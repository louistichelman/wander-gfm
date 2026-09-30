# Eval

`eval_one_dataset.sh` is the worker. `queue_eval_*.sh` submit one job per
dataset. Default checkpoint: `checkpoints/pretrained_wander/pretrained_wander.pt`.

```bash
DATASETS=CORA bash slurm_scripts/eval/queue_eval_nc.sh
```
