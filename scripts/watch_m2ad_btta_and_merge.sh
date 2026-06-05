#!/usr/bin/env bash
set -euo pipefail

root="${1:-/home/qilab/byeongju_lee/anomaly_tta/outputs/m2ad_rd4ad_btta_full_randomstream_seed0}"
shift || true
pids=("$@")
python_bin="${PYTHON_BIN:-/home/qilab/anaconda3/bin/python3}"

if [ "${#pids[@]}" -eq 0 ]; then
  echo "Usage: $0 <output_root> <pid> [<pid> ...]" >&2
  exit 2
fi

echo "[m2ad-watch] waiting pids=${pids[*]} root=${root} $(date -Is)"
while true; do
  alive=0
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      alive=1
    fi
  done
  complete=1
  for view in 000 030 060 090 120 150 180 210 240 270 300 330; do
    detail="${root}/view_${view}/robustad_detailed.csv"
    if [ ! -f "${detail}" ] || [ "$(wc -l < "${detail}")" -lt 21 ]; then
      complete=0
      break
    fi
  done
  [ "${alive}" -eq 0 ] && [ "${complete}" -eq 1 ] && break
  sleep 60
done

echo "[m2ad-watch] workers finished, merging $(date -Is)"
cd /home/qilab/byeongju_lee/anomaly_tta
"${python_bin}" scripts/merge_m2ad_btta_results.py "${root}"
echo "[m2ad-watch] merge done $(date -Is)"
