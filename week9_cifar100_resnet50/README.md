# Week 9 - CIFAR-100 ResNet-50, Forget 10 Classes

Goal: scale the dissertation experiments from CIFAR-10/ResNet-18 to
CIFAR-100/ResNet-50 and forget 10 out of 100 classes.

Default forget classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

This folder is isolated from previous weeks. It implements:

- Original CIFAR-100 ResNet-50 training
- Retrain baseline
- Fine-tune baseline
- Negative Gradient baseline
- DKF
- RA-DKF
- E-RA-DKF

## Smoke Check

Use this only to verify code paths:

```bash
cd /Users/sidhu/Zoro/MU/week9_cifar100_resnet50
../.venv/bin/python run_experiments.py --stages original --download --original-epochs 1 --max-eval-samples 512
```

## Suggested Real Workflow

Train original model:

```bash
cd /Users/sidhu/Zoro/MU/week9_cifar100_resnet50
../.venv/bin/python run_experiments.py --stages original --download --original-epochs 100
```

Run baselines:

```bash
../.venv/bin/python run_experiments.py --stages baselines --reuse-checkpoints --retrain-epochs 100 --unlearn-epochs 10
```

Run DKF / RA-DKF / E-RA-DKF:

```bash
../.venv/bin/python run_experiments.py --stages dkf radkf eradkf --reuse-checkpoints --student-epochs 5 --lambda-align 0.05 --lambda-forget 0.06
```

Run everything in one command, if you are willing to leave it running:

```bash
../.venv/bin/python run_experiments.py --stages all --download --original-epochs 100 --retrain-epochs 100 --unlearn-epochs 10 --student-epochs 5 --lambda-align 0.05 --lambda-forget 0.06
```

## Notes

- This is much larger than the CIFAR-10 setup. Full original + retrain +
  unlearning experiments can take hours even on MPS.
- Checkpoints are ignored by git and stored under `checkpoints/`.
- Results are saved to `results/week9_cifar100_resnet50_results.json`.
