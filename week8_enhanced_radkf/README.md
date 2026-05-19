# Week 8 - Enhanced RA-DKF Trials

This folder is isolated from `week6_7_novelty`. It reuses the existing Week 1
and Week 3 checkpoints but writes new checkpoints/results only here.

## Trial Ideas

- `avgpool`: cosine alignment on final ResNet avgpool features.
- `layer3`: cosine alignment on intermediate layer3 pooled features.
- `layer2-layer3`: cosine alignment on two mid-level layers.
- retain contrastive negatives are detached by default, so the contrastive loss
  does not directly move retain features.

## Commands

Quick smoke trial:

```bash
cd week8_enhanced_radkf
../.venv/bin/python run_experiments.py --variants avgpool --lambda-aligns 0.1 --student-epochs 1 --max-train-batches 5 --quick-eval --max-eval-samples 256
```

Full first sweep:

```bash
cd week8_enhanced_radkf
../.venv/bin/python run_experiments.py --variants avgpool layer3 layer2-layer3 --lambda-aligns 0.1 0.5 1.0
```

Evaluate saved Week 8 checkpoints:

```bash
cd week8_enhanced_radkf
../.venv/bin/python run_experiments.py --reuse-checkpoints
```

## Target

Beat base DKF on `Avg.Gap` or clearly reduce `Retain_Feature_Drift` without
hurting `MIA` and `Acc_Df`. A publishable result should show the trade-off, not
only one lucky metric.
