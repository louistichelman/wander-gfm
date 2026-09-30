# Pretrain ablations

Same stages as `pretrain/`, one hyperparameter changed.

```bash
bash slurm_scripts/submit.sh slurm_scripts/ablations/train_stage1_ablation_syn_link_pred_0.sh
```

GraphPFN / NodePFN prior scripts need extra packages in the wander env
(`requirements-priors.txt` or DGL; see those scripts).
