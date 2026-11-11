#!/bin/bash
#SBATCH --job-name=scaleproto-metric
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=metric-%j.out

set -euo pipefail
metric=${1:-}
case "$metric" in
  consistency|stability) shift ;;
  *) echo 'Usage: sbatch scripts/run_metric.slurm consistency|stability MODEL STAGE DATASET [OPTIONS]' >&2; exit 2 ;;
esac

repo_root=${SCALEPROTOSEG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$repo_root"
if [[ ! -d segmentation/analysis/metrics ]]; then
  echo 'Submit from the repository root or set SCALEPROTOSEG_ROOT.' >&2
  exit 2
fi

python=${METRIC_PYTHON:-"$PWD/.venv/bin/python"}
if [[ ! -x "$python" ]]; then
  echo 'Set METRIC_PYTHON to an executable Python interpreter or create .venv.' >&2
  exit 2
fi
exec "$python" -u -m "segmentation.analysis.metrics.$metric" "$@"
