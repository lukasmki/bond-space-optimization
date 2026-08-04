#!/bin/bash
# Example SLURM submission for the HCombustion pipeline.
#
#   mkdir -p data/logs                       # sbatch fails if the log dir is missing
#
#   sbatch data/run.slurm 0-prep.py          # stage 0: annotate frames with bond orders
#   sbatch --array=1-19 data/run.slurm 1-run.py   # stage 1: one reaction per array task
#   sbatch --time=12:00:00 data/run.slurm 2-network.py   # network discovery (serial)
#
# Submit from the repository root, not from data/.
#
# The three stages have very different shapes, so override the directives on
# the command line rather than keeping three near-identical copies of this
# file -- anything passed to sbatch wins over the #SBATCH lines below.

#SBATCH --job-name=bondspace
#SBATCH --output=data/logs/%x-%A_%a.out
#SBATCH --error=data/logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail

STAGE="${1:-1-run.py}"

# --- environment -----------------------------------------------------------
# Site-specific; adjust or delete.  The point is that `uv` and a compiler
# runtime providing OpenMP have to be on PATH.
# module load python/3.13
# module load gcc

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# --- threads ---------------------------------------------------------------
# configure_threads() in data/util.py sets PySCF's own OpenMP count at runtime
# and will pick SLURM_CPUS_PER_TASK up by itself.  BLAS is the reason to also
# export here: those libraries read their thread count once, when they are
# loaded, which happens before any of our code runs.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

# Keep threads on the cores actually allocated.
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# --- reaction selection (array jobs only) ----------------------------------
# 1-run.py processes every pending reaction unless BONDSPACE_RXN names one.
# Selecting is mandatory under an array: the skip-if-output-exists check only
# sees a *finished* reaction, so tasks launched together would otherwise all
# start on the same one.  Stages 0 and 2 ignore this and are submitted without
# --array.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    export BONDSPACE_RXN
    BONDSPACE_RXN="$(printf 'rxn_%02d' "$SLURM_ARRAY_TASK_ID")"
    echo "array task ${SLURM_ARRAY_TASK_ID} -> ${BONDSPACE_RXN}"
fi

# --- run -------------------------------------------------------------------
echo "host      : $(hostname)"
echo "stage     : ${STAGE}"
echo "cpus/task : ${SLURM_CPUS_PER_TASK:-unset}"
echo "started   : $(date -Is)"

uv run python "data/${STAGE}"

echo "finished  : $(date -Is)"
