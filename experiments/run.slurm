#!/bin/bash
# SLURM submission for the experiment suite.
#
#   mkdir -p experiments/logs                   # sbatch fails if the log dir is missing
#
# Job counts come from the registry, never hardcoded -- ask for them:
#
#   uv run python experiments/registry.py       # all counts
#   uv run python experiments/02_ts_single_ended.py --list
#
#   sbatch --array=0-18             experiments/run.slurm 01_reference_states.py
#   sbatch --array=0-18             experiments/run.slurm 02_ts_single_ended.py
#   sbatch --array=0-56             experiments/run.slurm 03_information_ladder.py
#   sbatch --array=0-132            experiments/run.slurm 04_ts_baselines.py
#   sbatch --array=0-111            experiments/run.slurm 05_target_sharpness.py
#   sbatch --array=0-75             experiments/run.slurm 06_path_vs_irc.py
#   sbatch --array=0-4 --time=24:00:00 experiments/run.slurm 07_network_discovery.py
#   sbatch --array=0-495%50         experiments/run.slurm 08_ablations.py
#   sbatch --array=0-21 --exclusive --cpus-per-task=16 experiments/run.slurm 09_scaling.py
#
# Submit from the repository root, not from experiments/.
#
# Run order: E01 -> E02 -> {E03, E04, E05} -> {E06, E07} -> E08 -> E09 -> E10.
# E01 is a hard prerequisite: every other TS experiment skips reactions whose
# reference could not be verified, so running them first produces an empty
# study rather than a wrong one.
#
# The stages have very different shapes, so override directives on the command
# line rather than keeping several near-identical copies of this file --
# anything passed to sbatch wins over the #SBATCH lines below.

#SBATCH --job-name=bondspace-exp
#SBATCH --output=experiments/logs/%x-%A_%a.out
#SBATCH --error=experiments/logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail

STAGE="${1:-02_ts_single_ended.py}"
shift || true

# --- environment -----------------------------------------------------------
# Site-specific; adjust or delete.  `uv` and a compiler runtime providing
# OpenMP have to be on PATH.
# module load python/3.13
# module load gcc

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# --- threads ---------------------------------------------------------------
# bondspace.threads.configure_threads sets PySCF's own OpenMP count at runtime
# and picks SLURM_CPUS_PER_TASK up by itself.  BLAS is the reason to also
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

# --- job selection ---------------------------------------------------------
# Unlike data/run.slurm, which maps the array index onto a reaction id, the
# index here is an index into the experiment's registry: most experiments have
# job counts unrelated to the 19 reactions (496 for the ablations).  Each
# script's --list prints its count so the array bound can be computed.
ARGS=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    ARGS+=(--index "${SLURM_ARRAY_TASK_ID}")
    echo "array task ${SLURM_ARRAY_TASK_ID}"
fi

# --- run -------------------------------------------------------------------
echo "host      : $(hostname)"
echo "stage     : ${STAGE}"
echo "cpus/task : ${SLURM_CPUS_PER_TASK:-unset}"
echo "started   : $(date -Is)"

uv run python "experiments/${STAGE}" "${ARGS[@]+"${ARGS[@]}"}" "$@"

echo "finished  : $(date -Is)"
