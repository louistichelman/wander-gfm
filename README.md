# Wander

Accompanying code for *To Learn is to Wander: Learning Across Graphs and Tasks with Random Walks*.

A single pretrained checkpoint covers node classification, homogeneous link prediction, and knowledge-graph link prediction. Evaluation datasets are downloaded into `raw_data/` on first use.

The checkpoint is not in the repo. Download it from
https://osf.io/2us6d/overview?view_only=a490e7fd33ef4ec2903cd2dbdbf5db15
and place it at `checkpoints/pretrained_wander/pretrained_wander.pt`.

## Install

```bash
conda create -n wander python=3.11 pip -y
conda activate wander
python -m pip install -r requirements.txt
mkdir -p logs raw_data checkpoints results/eval
```

`graph-walker` builds a C++ extension, so `g++` and Python headers must be available. Evaluation and training expect an NVIDIA GPU. Optional W&B: `WANDB_MODE=disabled`, or `wandb login`.

`submit.sh` uses `sbatch` when it exists and otherwise runs the job locally. Queue scripts (`queue_*.sh`) always run in the current shell and submit one job per dataset; do not `sbatch` them. Edit the `#SBATCH` headers for your site, or set `SLURM_PARTITION`, `SLURM_GRES`, `SLURM_NUM_GPUS`, `CLUSTER_MEM`, and `CONDA_ENV` (default `wander`). Append `--dry-run` to a queue script to print the jobs without launching them.

A small check:

```bash
bash slurm_scripts/submit.sh slurm_scripts/eval/eval_one_dataset.sh CORA
```

Per-dataset metrics land in `results/eval/<run>/<dataset>/metrics.json`. Each queue also writes `eval_summary.txt`.

## Paper tables

Dataset lists are in `slurm_scripts/paper_protocol.sh`. These commands reproduce the released checkpoint. Full queues are cluster jobs: the knowledge-graph suite is 54 graphs, and the larger node-classification graphs request 128–256G of host memory.

| Paper | Command |
|-------|---------|
| Tables 2 and 12, knowledge-graph link prediction | `bash slurm_scripts/eval/queue_eval_kg.sh` |
| Table 3, homogeneous link prediction | `bash slurm_scripts/eval/queue_eval_lp.sh` |
| Tables 4 and 19, node classification | `bash slurm_scripts/eval/queue_eval_nc.sh` |
| Table 6, features × relations | `bash slurm_scripts/eval/queue_eval_kg_composition.sh` |
| Table 7, long-range / no-feature node classification | `bash slurm_scripts/eval/queue_eval_nc_longrange.sh` |
| Table 16, UniLP comparison | `bash slurm_scripts/eval/queue_eval_unilp.sh` |
| Figure 2 and inference ablations | `slurm_scripts/eval/inference_ablations/` |

Restrict a queue with `DATASETS`, for example `DATASETS="CORA CITESEER" bash slurm_scripts/eval/queue_eval_nc.sh`. Node classification defaults to seeds `0:1:2` and every official split (`NC_SPLITS=first` keeps split 0 only).

Table 6 needs MiniLM features (`requirements-minilm.txt`). The structure-only cells of that table are `queue_eval_kg_composition.sh` with features and relations switched off; see the script header.

## Pretraining

The released checkpoint is the end of a four-stage curriculum (Table 8). Stages run in order, two GPUs, and stages 3–4 want about 256G of host memory:

```bash
bash slurm_scripts/submit.sh slurm_scripts/pretrain/train_stage1_baseline.sh
bash slurm_scripts/submit.sh slurm_scripts/pretrain/train_stage2_baseline.sh
bash slurm_scripts/submit.sh slurm_scripts/pretrain/train_stage3_baseline.sh
bash slurm_scripts/submit.sh slurm_scripts/pretrain/train_stage4_baseline.sh
```

Each stage resumes from the previous one. See `slurm_scripts/pretrain/README.md`.

Pretraining ablations (priors, no random-walk update, synthetic-only) are in `slurm_scripts/ablations/`. Finetuning (Tables 22–25) is in `slurm_scripts/finetune/`.

## Baselines

Not required for the Wander tables above.

```bash
bash scripts/setup_third_party.sh
bash baselines/setup_envs.sh
```

Pins are in `third_party/PINS`. Official baseline weights are listed in `third_party/README.md`. Launchers are in `slurm_scripts/baselines/`. NBFNet, BUDDY, link-prediction heuristics, and the node-classification GCN grid use the `wander` environment. GraphPFN and NodePFN prior ablations need `requirements-priors.txt`.
