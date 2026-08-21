# Segmentation Models

PyTorch model implementations used by the mineral pore segmentation experiments.

## Included Families

- `unet_model.py`: U-Net variants used as the main baseline family.
- `multiscale_attention_unet.py`: multi-scale attention U-Net variants.
- `segformer.py`: SegFormer-style segmentation model.
- `dinov2_unet.py` and `dinov3_unet.py`: foundation-model encoder experiments.
  The historical DINOv2 `torch.hub` loader is disabled in the public runtime:
  it neither downloads mutable repository code nor silently substitutes a mock
  model. Re-enabling it requires separately reviewed, revision-pinned local
  source and verified weights.
- `yolov8_seg.py`: YOLO-style segmentation experiment support.

## Expected Task

The current reproducible pipeline uses three output classes:

| Channel | Class |
| --- | --- |
| 0 | Disconnected pores |
| 1 | Connected pores |
| 2 | Mineral matrix |

Some older experiments and helper scripts used two pore-only classes. Check each saved `training_config.json` or `training_summary.json` before comparing metrics.

## Checkpoints

Do not commit model checkpoints directly unless they are intentionally released through Git LFS or an external archive. Local checkpoints are ignored by `.gitignore`.

For DINOv3 models, pass `pretrained_path` explicitly if a local checkpoint is available. The default is `None`, which falls back to random initialization for code compatibility.
