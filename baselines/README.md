# Baselines

```bash
bash scripts/setup_third_party.sh
bash baselines/setup_envs.sh          # or e.g. graphany_env
```

Set `TORCH_INDEX_URL` if you need a different PyTorch wheel. GraphPFN and
GraphAny also need DGL. Official weights go in the checkouts; see
`third_party/README.md`.

Launchers: `slurm_scripts/baselines/`. NBFNet, BUDDY, heuristics, and the
node-classification GCN grid use `wander`.
