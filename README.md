# Boundary Test-Time Adaptation for RD4AD Anomaly Detection

This folder contains the local assets, wrappers, and experiment outputs for the
current Boundary Test-Time Adaptation, BTTA, research direction on RD4AD-based
industrial anomaly detection.

The core implementation lives in:

- `/home/qilab/byeongju_lee/bt_recsvm_research/scripts/run_robustad_anomalib_rd4ad_bt_recsvm.py`

This folder keeps the data, pretrained checkpoints, M2AD wrappers, and result
reports under one path:

- `/home/qilab/byeongju_lee/anomaly_tta`

## Method Summary

BTTA adapts an RD4AD anomaly detector online at test time using a small active
label budget and a learned normal/anomaly boundary. The method is designed for
streaming target-domain data where the source model is fixed and the target
distribution may shift.

Current working variant:

- Backbone detector: RD4AD / Reverse Distillation with `wide_resnet50_2`
- Input resolution: `512 x 512`
- Test stream: random stream
- Active query: one sample per batch after bootstrap
- Boundary model: linear SVM
- Boundary features: 5D anomaly-map statistics
- Tail pseudo labels: stream-past lower/upper tail labels for boundary training
- Model adaptation: BN-only adaptation
- Adaptation samples: true active normal labels plus confident pseudo-normal samples
- Adaptation score source: adapted model with EMA safety scoring
- Learning rate: `0.02`
- Batch size: `8`
- Optimizer update: one optimizer step per adapted sample

The current 5D feature vector is:

| Feature | Meaning |
| --- | --- |
| `score_max` | Maximum value of the anomaly map; this is the classic RD4AD image score. |
| `score_mean` | Mean anomaly-map value; captures global score shift. |
| `score_std` | Standard deviation of the anomaly map; separates localized defects from diffuse changes. |
| `area_ratio` | Fraction of pixels above the source normal pixel threshold; estimates defect area. |
| `source_z_max` | Z-score of `score_max` relative to the source normal score distribution. |

This 5D setting is produced by the runner as:

```bash
--active-svm-feature-mode map_stats \
--active-svm-drop-features score_top1_mean score_top5_mean source_z_top5
```

## Online BTTA Procedure

For each incoming target batch:

1. Score the batch using the current adapted model, with EMA-smoothed scoring.
2. Build anomaly-map features for each sample.
3. Query one active label near the current learned boundary.
4. Add the queried label to the boundary training set.
5. Update the stream-past tail memory:
   - lower 1% score tail is pseudo-normal, PN
   - upper 1% score tail is pseudo-anomaly, PA
6. Add tail pseudo labels to boundary training only:
   - active true labels use weight `1.0`
   - lower-tail PN uses weight `0.5`
   - upper-tail PA uses weight `0.2`
7. Refit the boundary model online.
8. Select confident pseudo-normal samples under the learned boundary.
9. Adapt only BN parameters of the RD4AD model using selected normal-like
   samples.

Important constraint: pseudo labels are used to shape the boundary, but tail
pseudo labels are not directly used as adaptation samples.

## Why This Is Different

The method is not only confidence-threshold TTA. Its novelty is the online
coupling of three signals:

- Active boundary labels: sparse queried labels anchor the normal/anomaly split.
- Stream-past tail pseudo labels: history-only score extremes stabilize boundary
  learning without using future data.
- Safe normal-side adaptation: only normal-like samples update the detector,
  while anomalies mainly inform the boundary.

The boundary model decides which samples are safe enough for adaptation, and the
RD4AD model is adapted through BN parameters only. EMA scoring reduces the risk
that a single unstable adapted model state dominates subsequent selection.

## Local Assets

RobustAD data:

- `/home/qilab/byeongju_lee/anomaly_tta/data/robustad`
- Categories used in the current protocol: `MetalParts`, `PCB`, `PiledBags`

AeBAD data:

- `/home/qilab/byeongju_lee/anomaly_tta/data/AeBAD`
- Current comparison category and splits: `AeBAD_V`, `video1 video2 video3`

M2AD single-view illumination-shift data:

- `/home/qilab/byeongju_lee/anomaly_tta/data/m2ad/domain_shift/single_view_illum_shift`
- Current comparison views: `000 030 060 090 120 150 180 210 240 270 300 330`
- Current comparison categories: `Bird Car Cube Dice Doll Holder Motor Ring Teapot Tube`

RD4AD pretrained checkpoints:

- `/home/qilab/byeongju_lee/anomaly_tta/pretrained/rd4ad_robustad_anomalib_table2_seed0_20260506/rd4ad_checkpoints`

M2AD wrapper entrypoint:

- `/home/qilab/byeongju_lee/anomaly_tta/scripts/run_m2ad_anomalib_rd4ad_btta.py`

M2AD launch script:

- `/home/qilab/byeongju_lee/anomaly_tta/scripts/run_m2ad_rd4ad_btta_views.sh`

## RobustAD Reference Command

The runner expects checkpoints under the chosen output directory as
`rd4ad_checkpoints`. The command below links the fixed seed0 source checkpoint
and runs the current 5D stream-past-tail BTTA variant for one stream seed.

```bash
PY=/home/qilab/anaconda3/bin/python
SCRIPT=/home/qilab/byeongju_lee/bt_recsvm_research/scripts/run_robustad_anomalib_rd4ad_bt_recsvm.py
DATA=/home/qilab/byeongju_lee/anomaly_tta/data/robustad
CKPT=/home/qilab/byeongju_lee/anomaly_tta/pretrained/rd4ad_robustad_anomalib_table2_seed0_20260506/rd4ad_checkpoints
OUT=/home/qilab/byeongju_lee/anomaly_tta/outputs/btta_5d_streamtail1_seed0

mkdir -p "${OUT}"
ln -sfn "${CKPT}" "${OUT}/rd4ad_checkpoints"

CUDA_VISIBLE_DEVICES=0 "${PY}" "${SCRIPT}" \
  --data-root "${DATA}" \
  --output-root "${OUT}" \
  --categories MetalParts PCB PiledBags \
  --device cuda \
  --seed 0 \
  --stream-seed 0 \
  --stream-order random \
  --resize-size 512 \
  --crop-size 512 \
  --batch-size 8 \
  --train-batch-size 8 \
  --run-tta \
  --tta-method active_svm_boundary \
  --tta-param-scope bn_only \
  --tta-lr 0.02 \
  --tta-steps 1 \
  --tta-score-source adapted_ema \
  --tta-score-ema-decay 0.95 \
  --active-svm-feature-mode map_stats \
  --active-svm-drop-features score_top1_mean score_top5_mean source_z_top5 \
  --active-boundary-model linear_svm \
  --active-svm-confidence-threshold 0.10 \
  --active-svm-tail-scope stream_past \
  --active-svm-tail-pseudo-label-fraction 0.01 \
  --active-svm-lower-tail-pseudo-normal-weight 0.5 \
  --active-svm-upper-tail-pseudo-anomaly-weight 0.2 \
  --no-continue-on-error
```

For matched stream-seed sweeps, keep `--seed 0` fixed and vary only
`--stream-seed`.

## M2AD Wrapper

The M2AD wrapper imports the upstream RobustAD runner and swaps the dataset and
report hooks for M2AD single-view illumination-shift splits.

Default launch script:

```bash
cd /home/qilab/byeongju_lee/anomaly_tta
scripts/run_m2ad_rd4ad_btta_views.sh 0 0 1 2 3 4
```

The current M2AD script uses random stream, RD4AD, BN-only adaptation, adapted
EMA scoring, map-stat features, linear SVM boundary, and tail pseudo labels.

## Batch-Size 32 Comparison Queue

The active batch-size 32 comparison queue is:

- Output root: `/home/qilab/byeongju_lee/anomaly_tta/outputs/seed012_atta_eatta_btta_pseudo40_randomstream_512_bs32_decoder_bn_only_gpu1`
- Launcher: `/home/qilab/byeongju_lee/anomaly_tta/outputs/seed012_atta_eatta_btta_pseudo40_randomstream_512_bs32_decoder_bn_only_gpu1/run_seed012_bs32_gpu1_jobs.sh`
- Final summarizer: `/home/qilab/byeongju_lee/anomaly_tta/outputs/seed012_atta_eatta_btta_pseudo40_randomstream_512_bs32_decoder_bn_only_gpu1/summarize_seed012_bs32_gpu1.py`
- Final per-seed CSV: `/home/qilab/byeongju_lee/anomaly_tta/outputs/seed012_atta_eatta_btta_pseudo40_randomstream_512_bs32_decoder_bn_only_gpu1/summary_seed012_bs32_gpu1.csv`
- Final aggregate CSV: `/home/qilab/byeongju_lee/anomaly_tta/outputs/seed012_atta_eatta_btta_pseudo40_randomstream_512_bs32_decoder_bn_only_gpu1/aggregate_seed012_bs32_gpu1.csv`

Common matched settings:

- Physical GPU: `GPU1` only, via `CUDA_VISIBLE_DEVICES=1`
- Source checkpoint seed: fixed at `--seed 0`
- Stream seeds: `--stream-seed 0 1 2`
- Stream order: `random`
- Resolution: `512 x 512`
- Batch size: `32`
- Workers: `--num-workers 0`
- Adaptation scope: `decoder_bn_only`
- Learning rate and update count: `--tta-lr 0.02`, `--tta-steps 1`

Method mapping for this queue:

| Report label | RobustAD / AeBAD CLI method | M2AD CLI method | Key settings |
| --- | --- | --- | --- |
| `Base` | `source` rows from the BTTA group | `source` rows from the BTTA group | Fixed seed0 RD4AD source model |
| `ATTA` | `atta_paper_hybrid` | `atta_paper_hybrid` | `tta_score_source=adapted`, pseudo-normal entropy q `0.4`, max fraction `0.4`, replay `full` |
| `EATTA` | `eatta_paper_hybrid` | `eatta_paper` | `tta_score_source=adapted`, pseudo-normal entropy q `0.4`, max fraction `0.4`, replay `recent` |
| `BTTA` | `active_boundary` | `active_boundary` | `tta_score_source=adapted_ema`, `linear_svm`, `multiscale_frequency_nosource`, confidence `0.25`, tail pseudo-label fraction `0.0` |

Progress snapshot as of `2026-06-05T10:39:52+09:00`:

| Stream seed | RobustAD groups | AeBAD groups | M2AD view CSVs | M2AD merged summaries | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `0` | `3/3` | `3/3` | `36/36` | `3/3` | Complete |
| `1` | `3/3` | `3/3` | `34/36` | `2/3` | Running `M2AD / BTTA / view_270` |
| `2` | `0/3` | `0/3` | `0/36` | `0/3` | Pending |

Do not mix this batch-size 32 queue with the earlier batch-size 8 reports when
making the final table. The final aggregate CSV is the source of truth once all
three stream seeds finish.

## Main RobustAD Results

All values below are target split image AUROC, excluding source-domain `test0`.
The source RD4AD checkpoint is fixed to seed0; only `--stream-seed` varies over
`0, 1, 2`.

### Core Baselines

| Method | Target AUROC |
| --- | ---: |
| Source RD4AD | `71.59` |
| ATTA-AD | `77.23 +/- 1.10` |
| EATTA-AD | `76.45 +/- 1.39` |
| TENT-AD | `51.84 +/- 3.53` |
| SAR-AD | `58.97 +/- 3.01` |
| BTTA 5D asym-tail C, batch-tail 5% | `79.60 +/- 0.83` |
| BTTA 5D stream-past-tail 1% | `79.74 +/- 1.21` |

### A0-A3 Ablation

| Variant | Target AUROC | Delta vs Source |
| --- | ---: | ---: |
| Source RD4AD | `71.59 +/- 0.00` | `+0.00` |
| A0: 1D score + linear SVM + no tail | `78.45 +/- 1.80` | `+6.86` |
| A1: 1D score + linear SVM + stream-past tail 1% | `79.67 +/- 0.74` | `+8.08` |
| A2: 5D + linear SVM + no tail | `78.19 +/- 2.11` | `+6.60` |
| A3: 5D + linear SVM + stream-past tail 1% | `79.74 +/- 1.21` | `+8.15` |
| Label-only: 5D boundary query, adapt requested labels only | `76.90 +/- 0.44` | `+5.31` |

### 1D Boundary Model Sweep

All rows use 1D score feature plus stream-past tail 1%.

| Boundary model | Target AUROC | Delta vs 1D linear SVM |
| --- | ---: | ---: |
| Linear SVM | `79.67 +/- 0.74` | `+0.00` |
| Logistic regression | `79.65 +/- 0.81` | `-0.01` |
| Student-t density | `79.42 +/- 1.17` | `-0.25` |
| GDA / LDA | `79.34 +/- 1.28` | `-0.32` |
| KDE | `79.33 +/- 1.26` | `-0.34` |
| Isotonic + logistic | `79.10 +/- 1.17` | `-0.57` |
| QDA | `78.64 +/- 1.50` | `-1.03` |
| Class-constrained GMM | `77.91 +/- 0.81` | `-1.76` |

The current evidence supports keeping a linear SVM boundary. Logistic regression
is effectively tied in 1D, but it does not improve the current best setting.

## Result Reports

Key saved reports:

- `/home/qilab/byeongju_lee/anomaly_tta/outputs/active_svm_boundary_mapstats_bn_full_randomstream_seed0_bs8_conf0p10_lr2e2_tailpseudo5pct_adaptema95_gpu1/feature_analysis/atta_eatta_btta_core_seed012_report.md`
- `/home/qilab/byeongju_lee/anomaly_tta/outputs/active_svm_boundary_mapstats_bn_full_randomstream_seed0_bs8_conf0p10_lr2e2_tailpseudo5pct_adaptema95_gpu1/feature_analysis/robustad_tta_sota_seed012_report.md`
- `/home/qilab/byeongju_lee/anomaly_tta/outputs/active_svm_boundary_mapstats_bn_full_randomstream_seed0_bs8_conf0p10_lr2e2_tailpseudo5pct_adaptema95_gpu1/feature_analysis/robustad_btta_stream_past_tail1_seed012_report.md`
- `/home/qilab/byeongju_lee/anomaly_tta/outputs/active_svm_boundary_mapstats_bn_full_randomstream_seed0_bs8_conf0p10_lr2e2_tailpseudo5pct_adaptema95_gpu1/feature_analysis/robustad_btta_ablation_a0_a3_labelonly_seed012_report.md`
- `/home/qilab/byeongju_lee/anomaly_tta/outputs/active_svm_boundary_mapstats_bn_full_randomstream_seed0_bs8_conf0p10_lr2e2_tailpseudo5pct_adaptema95_gpu1/feature_analysis/robustad_1d_boundary_models_streamtail1_seed012_report.md`
- `/home/qilab/byeongju_lee/anomaly_tta/outputs/seed012_atta_eatta_btta_pseudo40_randomstream_512_bs32_decoder_bn_only_gpu1/aggregate_seed012_bs32_gpu1.csv`

## Naming

Use the method name:

```text
Boundary Test-Time Adaptation (BTTA)
```

A precise variant name for current RobustAD reporting is:

```text
BTTA-5D-SPTail1
```

where `5D` means the reduced map-stat feature vector and `SPTail1` means
stream-past lower/upper 1% tail pseudo labels.

## Fair Comparison Rules

When comparing methods in this project:

- Keep the RD4AD source checkpoint fixed unless retraining is explicitly part of
  the experiment.
- For stream-seed sweeps, keep `--seed 0` and vary only `--stream-seed`.
- Use random stream unless a specific sequential-stream experiment is requested.
- Keep LR, batch size, update count, score source, EMA, and adaptation parameter
  scope matched across TTA methods.
- Exclude `test0` when reporting target-domain RobustAD metrics.
- Do not report smoke or subset runs as full benchmark comparisons.
