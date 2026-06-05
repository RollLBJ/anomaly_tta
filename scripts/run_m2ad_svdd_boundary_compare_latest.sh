#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <gpu_id> [<view> ...]" >&2
  exit 2
fi

gpu_id="$1"
shift

if [ "$#" -gt 0 ]; then
  views=("$@")
else
  views=(000 030 060 090 120 150 180 210 240 270 300 330)
fi

python_bin="${PYTHON_BIN:-/home/qilab/anaconda3/bin/python}"
split_base="/home/qilab/byeongju_lee/anomaly_tta/data/m2ad/domain_shift/single_view_illum_shift"
output_base="/home/qilab/byeongju_lee/anomaly_tta/outputs/m2ad_btta_ce_all_active_pseudo_joint_svm_margin_seed0_randomstream_512_bs4_conf0p25_decoder_bn_only_gpu${gpu_id}"
checkpoint_base="/home/qilab/byeongju_lee/anomaly_tta/outputs/m2ad_rd4ad_btta_full_randomstream_seed0"
categories=(Bird Car Cube Dice Doll Holder Motor Ring Teapot Tube)

mkdir -p "${output_base}/logs"

for view in "${views[@]}"; do
  data_root="${split_base}/view_${view}_source_I01_targets_I02-I10"
  checkpoint_root="${checkpoint_base}/view_${view}/rd4ad_checkpoints"
  output_root="${output_base}/view_${view}"
  echo "[m2ad-latest] start view=${view} gpu=${gpu_id} data_root=${data_root} output_root=${output_root} $(date -Is)"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${python_bin}" scripts/run_m2ad_svdd_boundary_compare.py \
    --data-root "${data_root}" \
    --checkpoint-root "${checkpoint_root}" \
    --output-root "${output_root}" \
    --categories "${categories[@]}" \
    --splits test \
    --methods source active_boundary \
    --boundary-model linear_svm \
    --feature-family multiscale_frequency \
    --device cuda \
    --seed 0 \
    --stream-seed 0 \
    --stream-order random \
    --resize-size 512 \
    --crop-size 512 \
    --batch-size 4 \
    --num-workers 2 \
    --tta-param-scope decoder_bn_only \
    --tta-lr 0.02 \
    --tta-steps 1 \
    --active-svm-confidence-threshold 0.25 \
    --active-svm-tail-pseudo-label-fraction 0.01 \
    --active-svm-lower-tail-pseudo-normal-weight 1.0 \
    --active-svm-upper-tail-pseudo-anomaly-weight 1.0 \
    --active-label-ce-mode all \
    --active-label-ce-targets active_pseudo \
    --active-label-ce-weight 1.0 \
    --active-label-ce-update joint \
    --active-label-ce-pseudo-weight-mode svm_margin
  echo "[m2ad-latest] done view=${view} gpu=${gpu_id} $(date -Is)"
done

echo "[m2ad-latest] all done gpu=${gpu_id} views=${views[*]} $(date -Is)"
