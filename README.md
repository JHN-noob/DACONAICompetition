# Structure Stability Pipeline

This project is designed to run from the `dacon-ai-1/` root in Jupyter.

## Current Baseline

- Main baseline config: `configs/efficientnetv2_m_baseline.yaml`
- Backbone: `tf_efficientnetv2_m`
- Shared encoder with lightweight `front/top` view-specific adapters
- Fusion head uses `concat(front_adapter_feature, top_adapter_feature)`
- Train-time augmentation is synchronized across `front.png` and `top.png`
- Dev/Test preprocessing is fixed to `384x384 resize + normalize only`
- Validation structure is `train 5-fold + untouched external dev`
- EfficientNet uses `2 seeds`
- Per-model calibration is `temperature-only` on train OOF
- Final ensemble calibration is `bias-only` on external dev
- Default ensemble is `Efficient only`
- `stable(label=0)` hard samples: top `12%`, duplicated `4x`
- `unstable(label=1)` hard samples: top `5%`, duplicated `2x`
- It still starts from each fold's baseline best checkpoint and fine-tunes for `3` more epochs with `lr=3e-5`
- If the reference plain-M OOF does not exist at `outputs/oof/efficientnetv2_m_384_oof.csv`, `run_oof()` automatically creates it first and then continues the hard-finetune baseline

## Preprocessing

### Train only

- Resize to `384x384`
- Shared brightness/contrast augmentation across `front` and `top`
- Shared gamma augmentation across `front` and `top`
- Shared shadow/compression/blur/noise severity across `front` and `top`
- Shared affine augmentation across `front` and `top`

### Dev/Test only

- Resize to `384x384`
- Normalize only

## Notebook Order

### 0. Common import

```python
from src.common import load_config
```

### 1. Run train-only OOF

```python
from src.run_oof import run_oof

eff_cfg = load_config("configs/efficientnetv2_m_baseline.yaml")
print(run_oof(eff_cfg))
```

### 2. Run per-model calibration

```python
from src.calibrate import run_calibration

eff_cfg = load_config("configs/efficientnetv2_m_baseline.yaml")
print(run_calibration(eff_cfg))
```

### 3. Run ensemble selection and submission

```python
from src.ensemble import run_ensemble_oof, run_ensemble_submission

ensemble_cfg = load_config("configs/ensemble_baseline.yaml")
print(run_ensemble_oof(ensemble_cfg))
print(run_ensemble_submission(ensemble_cfg))
```

## Important Notes

- `dev.csv` is not part of fold construction.
- `run_oof()` saves:
  - train OOF
  - external dev mean predictions
  - test mean predictions
- `run_calibration()` calibrates on train OOF, not on dev.
- `run_ensemble_oof()` uses `ensemble.selection_split`, which is currently `dev`.
- `run_ensemble_submission()` prints external dev logloss from the saved `dev_path`.

## Output Paths

- Checkpoints: `outputs/checkpoints/`
- OOF: `outputs/oof/`
- Logs: `outputs/logs/`
- Calibration: `outputs/calibration/`
- Submissions: `outputs/submissions/`

Key files:

- `outputs/oof/efficientnetv2_m_384_oof.csv`
- `outputs/oof/efficientnetv2_m_384_baseline_oof.csv`
- `outputs/oof/efficientnetv2_m_384_baseline_dev_mean.csv`
- `outputs/oof/efficientnetv2_m_384_baseline_test_mean.csv`
- `outputs/calibration/efficientnetv2_m_384_baseline_calibration.json`
- `outputs/logs/efficientnetv2_m_384_baseline_hard_examples.csv`
- `outputs/submissions/ensemble_baseline_submission_YYYYMMDD_HHMMSS.csv`

## Metrics Schema

- History CSV:
  - `epoch, train_loss, valid_loss, valid_logloss, valid_accuracy, valid_auc, lr`
- OOF mean CSV:
  - `id, label, raw_logit, prob`
- OOF all CSV:
  - `id, label, fold, seed, model_name, raw_logit, prob`
- Dev/Test mean CSV:
  - `id, raw_logit, prob`
  - dev also includes `label`
- Calibration JSON:
  - `before_logloss, after_logloss, best_method, temperature, selection_split, selection_path`

## Example Console Log

```text
[train] efficientnetv2_m_384_seed42_fold0 epoch=3/15 train_loss=0.1542 valid_loss=0.4311 valid_logloss=0.4308 valid_acc=0.8500 lr=1.00e-04
```
