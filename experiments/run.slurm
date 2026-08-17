#!/bin/bash -l
# SLURM submission for the experiment suite -- NERSC Perlmutter (CPU nodes).
#
# EDIT BEFORE FIRST USE: the --account line below.  Perlmutter rejects jobs
# that do not name a project ("Job request does not match any supported
# policy"), unless you have set a default project in Iris.
#
#   mkdir -p experiments/logs                   # sbatch fails if the log dir is missing
#
# Run the repository from $SCRATCH (/pscratch/sd/<u>/<user>/...), not from
# $HOME.  $HOME is not built for the metadata traffic a 496-task array makes,
# and the log paths below are relative to the submission directory.
#
# Job counts come from the registry, never hardcoded -- ask for them:
#
#   uv run python experiments/registry.py       # all counts
#   uv run python experiments/02_ts_single_ended.py --list
#
#   sbatch --array=0-18              experiments/run.slurm 01_reference_states.py
#   sbatch --array=0-18              experiments/run.slurm 02_ts_single_ended.py
#   sbatch --array=0-56              experiments/run.slurm 03_information_ladder.py
#   sbatch --array=0-132             experiments/run.slurm 04_ts_baselines.py
#   sbatch --array=0-97              experiments/run.slurm 05_target_sharpness.py
#   sbatch --array=0-75              experiments/run.slurm 06_path_vs_irc.py
#   sbatch --array=0-4 --time=24:00:00 experiments/run.slurm 07_network_discovery.py
#   sbatch --array=0-433%50          experiments/run.slurm 08_ablations.py
#   sbatch --array=0-21 -q regular -N 1 -c 256 experiments/run.slurm 09_scaling.py
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
#
# --- Perlmutter specifics you have to keep in mind ------------------------
#
# QOS:   "debug" is the DEFAULT on Perlmutter (max 30 min, 5 queued jobs), so
#        -q must always be set or the arrays will be rejected/truncated.
#        shared  -- up to half a node, up to 48 h, charged only for the
#                   fraction of the node used.  This is what the per-task
#                   stages want: 8 cores out of 128 in "regular" would
#                   otherwise be charged as a whole node-hour.
#        regular -- whole nodes, exclusive, up to 48 h.  Use for E09.
#        preempt -- half price on CPU after the first 2 h, but preemptible;
#                   worth it for the 496-task ablation if the stage can
#                   checkpoint/resume (add --requeue).
#
# Cores: a Perlmutter CPU node is 2 x AMD EPYC 7763 = 128 physical cores /
#        256 hardware threads, 512 GB.  Slurm counts -c in *hardware threads*,
#        so -c 16 is 8 physical cores.  OMP_NUM_THREADS is therefore set to
#        SLURM_CPUS_PER_TASK/2 below -- do not copy the old
#        OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK line, it doubles the thread
#        count onto the same cores and costs ~2x in wall time for PySCF.
#        The "shared" QOS caps at half a node: -c 128 (64 physical cores).
#
# Memory: in "shared", memory scales with the CPUs requested (roughly 2 GB per
#        hardware thread), so -c 16 already lands near 32 GB and --mem is not
#        needed.  Do NOT leave a --mem=16G directive in place for a "regular"
#        job: it would cap a whole 512 GB node at 16 GB.  Add --mem on the
#        command line only for shared-QOS stages that need more than default.

#SBATCH --account=m2834             # <-- EDIT: your NERSC project
#SBATCH --constraint=cpu            # required: CPU nodes, CPU allocation
#SBATCH --qos=shared                # required: "debug" is the default otherwise
#SBATCH --job-name=bondspace-exp
#SBATCH --output=experiments/logs/%x-%A_%a.out
#SBATCH --error=experiments/logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # 16 hardware threads = 8 physical cores
#SBATCH --time=08:00:00
##SBATCH --licenses=scratch         # uncomment to hold jobs when scratch is down
##SBATCH --requeue                  # pair with -q preempt

set -euo pipefail

STAGE="${1:-02_ts_single_ended.py}"
shift || true

# --- environment -----------------------------------------------------------
# "module load cpu" makes sure the CPU software stack is active even if the
# submitting shell had the GPU stack loaded.  PrgEnv-gnu is the Perlmutter
# default and provides the OpenMP runtime.
module load cpu
module load PrgEnv-gnu

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source venv/bin/activate

# Keep uv's cache and virtualenv off $HOME.  A conda/venv tree on
# /global/common/software is mounted read-only on the compute nodes and is the
# fastest place for the environment; $SCRATCH works but gets purged after
# 8 weeks of inactivity.
# export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH}/.cache/uv}"
# export UV_PROJECT_ENVIRONMENT=/global/common/software/<project>/bondspace-venv

# Run `uv sync` ONCE on a login node before submitting.  Letting 500 array
# tasks each resolve and materialise the environment at start-up hammers the
# filesystem and can fail on lock contention; --no-sync makes each task assume
# the environment is already there.
# UV_RUN_FLAGS=(--no-sync)

# --- threads ---------------------------------------------------------------
# bondspace.threads.configure_threads sets PySCF's own OpenMP count at runtime
# and picks SLURM_CPUS_PER_TASK up by itself.  BLAS is the reason to also
# export here: those libraries read their thread count once, when they are
# loaded, which happens before any of our code runs.
#
# Perlmutter is 2-way SMT, so half of SLURM_CPUS_PER_TASK is the physical core
# count.  BONDSPACE_OMP_THREADS overrides this, which is what E09 needs when
# it sweeps thread counts inside a full-node allocation.
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-2}"
PHYSICAL_CORES=$(( CPUS_PER_TASK / 2 ))
if (( PHYSICAL_CORES < 1 )); then
    PHYSICAL_CORES=1
fi

export OMP_NUM_THREADS="${BONDSPACE_OMP_THREADS:-${PHYSICAL_CORES}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

# MKL's Zen dispatch: without this, MKL selects a pre-AVX2 kernel on EPYC.
# Harmless if the stack uses OpenBLAS instead.
export MKL_DEBUG_CPU_TYPE=5

# Keep threads on the cores actually allocated.  SLURM_CPU_BIND is the
# NERSC-recommended way to get the same binding out of srun.
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export SLURM_CPU_BIND=cores

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
echo "qos       : ${SLURM_JOB_QOS:-unset}"
echo "cpus/task : ${CPUS_PER_TASK} threads (${PHYSICAL_CORES} physical cores)"
echo "omp       : ${OMP_NUM_THREADS}"
echo "started   : $(date -Is)"

# srun rather than a bare invocation: it is what applies the CPU binding to
# the allocated cores.  -c has to be repeated explicitly -- recent Slurm no
# longer propagates --cpus-per-task from the batch script into the step.
srun --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --cpu-bind=cores \
    python "experiments/${STAGE}" "${ARGS[@]+"${ARGS[@]}"}" "$@"

echo "finished  : $(date -Is)"

