# Anomaly TTA Local Assets

Created: 2026-05-22

## Dataset

- Target: `/home/qilab/byeongju_lee/anomaly_tta/data/robustad`
- Source: `/home/qilab/byeongju_lee/bt_recsvm_research/data/robustad`
- Copy mode: hardlink copy via `cp -al`, so the original path is preserved and disk blocks are not duplicated.
- File count: 16,135
- Categories: `MetalParts`, `PCB`, `PiledBags`

Metadata:

- Target: `/home/qilab/byeongju_lee/anomaly_tta/data/robustad_meta`
- Source: `/home/qilab/byeongju_lee/bt_recsvm_research/data/robustad_meta`

## RD4AD RobustAD Pretrained Checkpoints

- Target: `/home/qilab/byeongju_lee/anomaly_tta/pretrained/rd4ad_robustad_anomalib_table2_seed0_20260506/rd4ad_checkpoints`
- Source: `/home/qilab/byeongju_lee/bt_recsvm_research/outputs/robustad_anomalib_rd4ad_table2_seed0_20260506/rd4ad_checkpoints`
- Copy mode: hardlink copy via `cp -al`.
- Checkpoint files:
  - `MetalParts/anomalib_reverse_distillation.pt`
  - `PCB/anomalib_reverse_distillation.pt`
  - `PiledBags/anomalib_reverse_distillation.pt`

