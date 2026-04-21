# DACON 스마트 물류창고 출고 지연 예측 대회 최종 제출 파이프라인

## 요약

이 저장소는 물류/창고 운영 데이터를 사용해 `avg_delay_minutes_next_30m`를 예측하는 회귀 파이프라인입니다. 최종 제출 모델은 `blend_v053_v047_stack_v006_diversity`이며, public score는 `10.0039760874`입니다.

최종 모델은 단일 모델 하나가 아니라 다음 두 예측을 섞은 앙상블입니다.

- `blend_v047_residual_v029_shrinkage`: 강한 residual 모델 `v029`를 단일 baseline `v021` 방향으로 보수적으로 shrink한 예측
- `stack_lgb_meta_v006_strong_family_context_aware`: 강한 모델군의 예측값, 예측 통계, 예측 차이, layout/context feature를 사용하는 LightGBM meta stack

최종 blend는 `v047`과 `stack_v006`의 OOF 기준 최적 조합이며, 선택 weight는 `v047 0.69 / stack_v006 0.31`입니다.

## 최종 성능

| run | 역할 | local OOF MAE | public score |
| --- | --- | ---: | ---: |
| `lightgbm_v021_no_layout_id_core_holdout_log1p_scenario_relative` | 강한 단일 baseline | `8.634888` | - |
| `lightgbm_v029_residual_v021_context_gate` | v021 기반 residual correction | `8.584945` | `10.0071030226` |
| `blend_v047_residual_v029_shrinkage` | v029와 v021 보수적 blend | `8.581921` | `10.0053310084` |
| `stack_lgb_meta_v006_strong_family_context_aware` | context-aware meta stack | `8.594994` | - |
| `blend_v053_v047_stack_v006_diversity` | 최종 제출 모델 | `8.578563` | `10.0039760874` |

## 프로젝트 구조

```text
.
├── configs/
│   ├── lightgbm_v013_no_layout_id_core_holdout_log1p.yaml
│   ├── lightgbm_v014_no_layout_id_core_holdout_tuned_log1p.yaml
│   ├── lightgbm_v015_no_layout_id_core_holdout_log1p_adv_weighted.yaml
│   ├── stack_lgb_meta_v003_log1p_family.yaml
│   ├── lightgbm_v021_no_layout_id_core_holdout_log1p_scenario_relative.yaml
│   ├── stack_lgb_meta_v005_context_aware.yaml
│   ├── lightgbm_v022_residual_stack_v005_context_gate.yaml
│   ├── lightgbm_v029_residual_v021_context_gate.yaml
│   ├── blend_v047_residual_v029_shrinkage.yaml
│   ├── stack_lgb_meta_v006_strong_family_context_aware.yaml
│   └── blend_v053_v047_stack_v006_diversity.yaml
├── data/
├── outputs/
├── src/
│   ├── final_pipeline.py
│   ├── run_pipeline.py
│   ├── train.py
│   ├── features.py
│   ├── residual_modeling.py
│   ├── stacking.py
│   ├── ensemble.py
│   └── ...
└── README.md
```

## 데이터 배치

노트북이나 스크립트는 프로젝트 루트에서 실행한다고 가정합니다. 아래 파일을 `data/`에 배치합니다.

```text
data/
├── train.csv
├── test.csv
├── layout_info.csv
└── sample_submission.csv
```

## 실행 환경 가정

이 저장소는 주피터 노트북 환경에서 실행하는 것을 전제로 정리했습니다. 주요 의존성은 `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `pyyaml`입니다.

## 최종 제출 파일 생성

처음부터 생성하는 경우:

```python
from src.final_pipeline import run_final_pipeline

submission_path = run_final_pipeline()
submission_path
```

필요한 중간 OOF/test prediction이 이미 `outputs/oof/`에 있는 경우:

```python
from src.final_pipeline import run_final_submission_from_existing_predictions

submission_path = run_final_submission_from_existing_predictions()
submission_path
```

생성되는 최종 제출 파일:

```text
outputs/submissions/blend_v053_v047_stack_v006_diversity_submission.csv
```

## 최종 파이프라인 실행 순서

`src/final_pipeline.py`의 `FINAL_RUN_ORDER`는 아래 순서로 고정되어 있습니다.

1. `lightgbm_v013_no_layout_id_core_holdout_log1p`
2. `lightgbm_v014_no_layout_id_core_holdout_tuned_log1p`
3. `lightgbm_v015_no_layout_id_core_holdout_log1p_adv_weighted`
4. `stack_lgb_meta_v003_log1p_family`
5. `lightgbm_v021_no_layout_id_core_holdout_log1p_scenario_relative`
6. `stack_lgb_meta_v005_context_aware`
7. `lightgbm_v022_residual_stack_v005_context_gate`
8. `lightgbm_v029_residual_v021_context_gate`
9. `blend_v047_residual_v029_shrinkage`
10. `stack_lgb_meta_v006_strong_family_context_aware`
11. `blend_v053_v047_stack_v006_diversity`

## 모델 설명

### 1. 기본 LightGBM 계열

`v013`, `v014`, `v015`, `v021`은 모두 LightGBM 기반입니다. 공통적으로 `layout_holdout` 5-fold CV를 사용하고, 주요 feature는 다음과 같습니다.

- 원본 수치 feature
- `layout_info.csv`에서 가져온 layout feature
- layout과 운영량의 ratio/cross feature
- scenario 단위 mean/std/min/max
- row-level 통계 feature
- `v021`에서는 scenario 내부 상대 위치 feature까지 사용

`v015`는 train/test 분포 차이를 반영하기 위해 adversarial domain classifier 기반 sample weight를 사용합니다.

### 2. Stacking 계열

`stack_v003`은 `v013/v014/v015`의 OOF/test prediction을 meta feature로 사용합니다. `stack_v005`와 `stack_v006`은 단순 예측값뿐 아니라 예측 평균, 표준편차, min/max, pairwise diff, `layout_type`, 핵심 context feature를 함께 사용합니다.

### 3. Residual correction 계열

`v022`는 `stack_v005`를 base prediction으로 두고 residual만 학습합니다. `v029`는 더 안정적인 `v021`을 base prediction으로 두고, `v022`, `stack_v005`, `stack_v003`, `v013`의 예측값과 예측 차이를 feature로 사용해 residual correction을 학습합니다.

### 4. 최종 blend

`v047`은 `v029`의 residual correction이 public에서 과해질 가능성을 줄이기 위해 `v021` 방향으로 shrink한 blend입니다. 최종 `v053`은 public에서 검증된 `v047`을 anchor로 두고, 다른 오류 패턴을 가진 `stack_v006`을 31% 반영한 diversity blend입니다.
