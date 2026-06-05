#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <gpu_id> <view> [<view> ...]" >&2
  exit 2
fi

gpu_id="$1"
shift

python_bin="${PYTHON_BIN:-/home/qilab/anaconda3/bin/python3}"
split_base="/home/qilab/byeongju_lee/anomaly_tta/data/m2ad/domain_shift/single_view_illum_shift"
output_base="/home/qilab/byeongju_lee/anomaly_tta/outputs/m2ad_rd4ad_btta_full_randomstream_seed0"
categories=(Bird Car Cube Dice Doll Holder Motor Ring Teapot Tube)

mkdir -p "${output_base}/logs"

for view in "$@"; do
  data_root="${split_base}/view_${view}_source_I01_targets_I02-I10"
  output_root="${output_base}/view_${view}"
  echo "[m2ad-btta] start view=${view} gpu=${gpu_id} data_root=${data_root} output_root=${output_root} $(date -Is)"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${python_bin}" scripts/run_m2ad_anomalib_rd4ad_btta.py \
    --data-root "${data_root}" \
    --output-root "${output_root}" \
    --categories "${categories[@]}" \
    --device cuda \
    --seed 0 \
    --stream-order random \
    --resize-size 512 \
    --crop-size 512 \
    --batch-size 8 \
    --train-batch-size 8 \
    --num-workers 2 \
    --rd4ad-epochs 200 \
    --run-tta \
    --tta-method active_svm_boundary \
    --tta-param-scope bn_only \
    --tta-lr 2e-2 \
    --tta-steps 10 \
    --tta-adapt-batch-size 8 \
    --tta-score-source adapted_ema \
    --tta-score-ema-decay 0.95 \
    --active-svm-feature-mode map_stats \
    --active-boundary-model linear_svm \
    --active-svm-confidence-threshold 0.10 \
    --active-svm-tail-pseudo-label-fraction 0.05 \
    --selector-mode expblend
  echo "[m2ad-btta] done view=${view} gpu=${gpu_id} $(date -Is)"
done

echo "[m2ad-btta] all done gpu=${gpu_id} views=$* $(date -Is)"
