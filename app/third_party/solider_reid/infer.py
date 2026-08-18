"""Inference-only SOLIDER ReID (Swin + BNNeck), no yacs/mmcv/train deps."""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbones.swin_transformer import (
    swin_base_patch4_window7_224,
    swin_small_patch4_window7_224,
    swin_tiny_patch4_window7_224,
)

_FACTORY = {
    "swin_base_patch4_window7_224": swin_base_patch4_window7_224,
    "swin_small_patch4_window7_224": swin_small_patch4_window7_224,
    "swin_tiny_patch4_window7_224": swin_tiny_patch4_window7_224,
}


def _weights_init_kaiming(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def _weights_init_classifier(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class SoliderReidModel(nn.Module):
    """build_transformer subset: backbone + BNNeck, eval returns pre-BN feat."""

    def __init__(
        self,
        transformer_type: str = "swin_small_patch4_window7_224",
        img_size: tuple[int, int] = (384, 128),
        num_classes: int = 1041,
        semantic_weight: float = 0.2,
        drop_path_rate: float = 0.1,
        neck_feat: str = "before",
    ) -> None:
        super().__init__()
        if transformer_type not in _FACTORY:
            raise ValueError(f"unknown SOLIDER transformer: {transformer_type}")
        factory = _FACTORY[transformer_type]
        self.neck_feat = neck_feat
        self.base = factory(
            img_size=list(img_size),
            drop_path_rate=drop_path_rate,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            pretrained=None,
            convert_weights=False,
            semantic_weight=semantic_weight,
        )
        self.in_planes = int(self.base.num_features[-1])
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(_weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(_weights_init_kaiming)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_feat, _featmaps = self.base(x)
        feat = self.bottleneck(global_feat)
        if self.training:
            return self.classifier(self.dropout(feat)), global_feat
        if self.neck_feat == "after":
            return feat
        return global_feat

    def load_param(self, trained_path: str) -> None:
        param_dict = torch.load(trained_path, map_location="cpu", weights_only=False)
        if isinstance(param_dict, dict) and "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]
        if isinstance(param_dict, dict) and "model" in param_dict and not any(
            k.startswith("base.") or k.startswith("bottleneck") for k in param_dict
        ):
            # unlikely wrapper; keep as-is if keys look like modules
            pass
        loaded, skipped = 0, 0
        own = self.state_dict()
        for key, value in param_dict.items():
            name = key.replace("module.", "")
            if name not in own:
                skipped += 1
                continue
            if own[name].shape != value.shape:
                skipped += 1
                continue
            own[name].copy_(value)
            loaded += 1
        if loaded == 0:
            raise RuntimeError(f"SOLIDER: no weights loaded from {trained_path}")


def build_solider_reid(
    weights: str,
    *,
    transformer_type: str = "swin_small_patch4_window7_224",
    image_size: tuple[int, int] = (384, 128),
    semantic_weight: float = 0.2,
    device: str = "cpu",
    num_classes: int = 1041,
) -> SoliderReidModel:
    model = SoliderReidModel(
        transformer_type=transformer_type,
        img_size=image_size,
        num_classes=num_classes,
        semantic_weight=semantic_weight,
        neck_feat="before",
    )
    model.load_param(weights)
    model.eval()
    model.to(device)
    return model
