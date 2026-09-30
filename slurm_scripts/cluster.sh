#!/usr/bin/env bash
# Optional Slurm overrides. Job scripts already contain #SBATCH headers;
# submit.sh only adds a flag when the matching environment variable is set.
#
#   SLURM_PARTITION   SLURM_QOS   SLURM_GRES   SLURM_NUM_GPUS
#   CLUSTER_MEM       CLUSTER_CPUS
#   CLUSTER_TIME_EVAL / CLUSTER_TIME_TRAIN / CLUSTER_TIME_FINETUNE
#   SLURM_JOB_NAME    SLURM_DEPENDENCY    SLURM_EXCLUDE_NODES
#
# Leave these unset to use the #SBATCH lines in the job script.

SLURM_NUM_GPUS="${SLURM_NUM_GPUS:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_NUM_GPUS}}"

cluster_sbatch_flags() {
  local kind="${1:-train}"
  [[ -n "${SLURM_PARTITION:-}" ]] && printf '%s\n' "--partition=${SLURM_PARTITION}"
  [[ -n "${SLURM_QOS:-}" ]] && printf '%s\n' "--qos=${SLURM_QOS}"
  [[ -n "${SLURM_GRES:-}" ]] && printf '%s\n' "--gres=${SLURM_GRES}"
  [[ -n "${CLUSTER_CPUS:-}" ]] && printf '%s\n' "--cpus-per-task=${CLUSTER_CPUS}"
  [[ -n "${CLUSTER_MEM:-}" ]] && printf '%s\n' "--mem=${CLUSTER_MEM}"
  local time_limit=""
  case "${kind}" in
    eval) time_limit="${CLUSTER_TIME_EVAL:-}" ;;
    finetune) time_limit="${CLUSTER_TIME_FINETUNE:-}" ;;
    tabular|train|*) time_limit="${CLUSTER_TIME_TRAIN:-}" ;;
  esac
  [[ -n "${time_limit}" ]] && printf '%s\n' "--time=${time_limit}"
  [[ -n "${SLURM_EXCLUDE_NODES:-}" ]] && printf '%s\n' "--exclude=${SLURM_EXCLUDE_NODES}"
  [[ -n "${SLURM_WANT_NODES:-}" ]] && printf '%s\n' "--nodelist=${SLURM_WANT_NODES}"
}

cluster_job_kind() {
  local base
  base="$(basename "$1")"
  case "${base}" in
    eval_*.sh|*_eval.sh|nodepfn_eval.sh|tabpfn_eval.sh|graphany_eval.sh|graphpfn_icl.sh|anygraph_eval.sh|anygraph_nc.sh|unilp_eval.sh|run_lp_heuristics.sh)
      echo eval ;;
    finetune_*.sh|run_nbfnet_baselines.sh|train_gcn_grid.sh|graphpfn_finetune.sh|run_lp_grid.sh|run_buddy_baselines.sh)
      echo finetune ;;
    *) echo train ;;
  esac
}
