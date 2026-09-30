# Job scripts

Wrappers around `main.py`. Edit `#SBATCH` headers for your site.
`submit.sh` uses `sbatch` when available, otherwise runs locally.

```bash
bash slurm_scripts/submit.sh slurm_scripts/eval/eval_one_dataset.sh CORA
DATASETS=CORA bash slurm_scripts/eval/queue_eval_nc.sh
```

Optional: `SLURM_PARTITION`, `SLURM_GRES`, `SLURM_NUM_GPUS`, `CLUSTER_MEM`,
`CONDA_ENV`. Do not `sbatch` `queue_*.sh` coordinators.

| Path | Role |
|------|------|
| `pretrain/` | PT1–PT4 |
| `ablations/` | Pretrain ablations |
| `eval/` | Zero-shot queues |
| `finetune/` | Finetune |
| `baselines/` | External baselines |
| `kg_protocol.sh` | KG walk / ensemble budgets |
| `paper_protocol.sh` | Dataset lists |

Default checkpoint: `checkpoints/pretrained_wander/pretrained_wander.pt`.
