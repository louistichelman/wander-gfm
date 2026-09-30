# Third-party checkouts

Not needed for Wander eval / pretrain / finetune.

```bash
bash scripts/setup_third_party.sh
```

Pins: `third_party/PINS`. Official weights (not in the zip):

| Baseline | Path |
|----------|------|
| GraphPFN | `graphpfn/checkpoints/graphpfn-adapters-1_3.pt` (or HF in TOML) |
| GraphAny | `GraphAny/checkpoints/graph_any_*.pt` |
| NodePFN | `NodePFN/models_ckpts/nodepfn/checkpoint_epoch_30.ckpt` |
| AnyGraph | `AnyGraph/Models/pretrain_link{1,2}.mod` |
| UniLP | `context_LP/checkpoints/pretrained/model.pt` |
