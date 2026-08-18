"""Minimal SOLIDER-REID inference package (Swin backbone + BNNeck).

Adapted from https://github.com/tinyvision/SOLIDER-REID (Apache/MIT — see LICENSE).
Train scripts and unused backbones are omitted; only embed() path is kept.
"""

from __future__ import annotations

from .infer import SoliderReidModel, build_solider_reid

__all__ = ["SoliderReidModel", "build_solider_reid"]
