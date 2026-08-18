# SOLIDER-REID (inference subset)

Sparse copy of [tinyvision/SOLIDER-REID](https://github.com/tinyvision/SOLIDER-REID) for offline crop embeddings.

- Backbone: `backbones/swin_transformer.py` (mmcv/cv2 imports removed; `.cuda()` → device of input)
- Wrapper: `infer.py` (`SoliderReidModel` + `build_solider_reid`)

Training scripts are not included. Download MSMT17 finetuned weights separately into `data/models/`.

See upstream LICENSE in this directory.
