#!/bin/bash
#SBATCH --job-name=scaleproto-metric
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/data_ssd/cluster_user_data/annamari/ScaleProtoSeg
#SBATCH --output=metric-%j.out

set -euo pipefail
metric=${1:-}
case "$metric" in
  consistency|stability) shift ;;
  *) echo 'Usage: sbatch scripts/run_metric.slurm consistency|stability MODEL STAGE DATASET [OPTIONS]' >&2; exit 2 ;;
esac
exec .venv-eval/bin/python -u -m "segmentation.analysis.metrics.$metric" "$@"
