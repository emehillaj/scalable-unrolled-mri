#!/bin/bash
# Multi-GPU training on one SLURM node: a single task starts one train.py process per GPU
# with torchrun. Submit from the repository root; every argument goes to train.py:
#
#   sbatch scripts/slurm_multigpu.sh --config configs/brain/ours.json
#   sbatch scripts/slurm_multigpu.sh --config configs/brain/ours.json --resume
#   sbatch --gpus-per-node=8 --time=24:00:00 scripts/slurm_multigpu.sh --config configs/brain/ours.json
#
# sbatch options go BEFORE the script name: anything after it is passed to train.py.
# effective_batch_size (8 in the configs) must be divisible by the number of GPUs.
# A run longer than the time limit is continued by a follow-up job with --resume, e.g.
#   sbatch --dependency=afterany:<JOBID> scripts/slurm_multigpu.sh --config ... --resume
#
# Optional environment variables (sbatch exports the submitting shell's environment):
#   CONDA_ENV  conda environment to activate in the job (default: use the current one)
#   NGPUS      number of processes (default: SLURM_GPUS_ON_NODE)
#
#SBATCH --job-name=unrolled-chains
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
##SBATCH --account=<ACCOUNT>
##SBATCH --partition=<PARTITION>
# --cpus-per-task: (num_workers + 1) per GPU. To change the GPU count or type, pass
# --gpus-per-node=N (or TYPE:N) before the script name. Do not combine it with --gres=gpu:...,
# which SLURM rejects next to this directive. On SLURM versions without --gpus-per-node,
# replace the directive with #SBATCH --gres=gpu:4.

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "usage: sbatch scripts/slurm_multigpu.sh --config CONFIG [train.py options...]" >&2
    exit 2
fi

cd "${SLURM_SUBMIT_DIR:-.}"
if [[ ! -f train.py ]]; then
    echo "no train.py in $PWD: submit from the repository root" >&2
    exit 1
fi

if [[ -n "${CONDA_ENV:-}" ]]; then
    # shellcheck source=/dev/null
    conda_base="$(conda info --base)"
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

NGPUS="${NGPUS:-${SLURM_GPUS_ON_NODE:-}}"
if [[ -z "$NGPUS" ]]; then
    python_cmd="${PYTHON:-python3}"
    if ! command -v "$python_cmd" >/dev/null 2>&1; then
        echo "cannot determine GPU count: $python_cmd is not available; set NGPUS explicitly" >&2
        exit 1
    fi
    NGPUS=$("$python_cmd" -c "import torch; print(torch.cuda.device_count())")
fi
if [[ ! "$NGPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NGPUS must be a positive integer, got: $NGPUS" >&2
    exit 2
fi
if ! command -v torchrun >/dev/null 2>&1; then
    echo "torchrun was not found in the job environment" >&2
    exit 1
fi

printf 'job %s on %s: %s processes, train.py' \
    "${SLURM_JOB_ID:-none}" "$(hostname)" "$NGPUS"
printf ' %q' "$@"
printf '\n'
exec torchrun --standalone --nnodes=1 --nproc_per_node="$NGPUS" train.py "$@"
