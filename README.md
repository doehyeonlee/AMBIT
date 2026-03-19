# AMBIT: Assessing Model Bias in Intersectional Traits

교차 정체성(Intersectionality)이 LLM의 편향에 미치는 영향을 3가지 맥락에서 측정하고, SAE(Sparse Autoencoder)를 통해 내부 표현의 합성 실패(compositional collapse)를 분석합니다.

---

## Quick Start (결과 재현)

데이터와 실험이 완료된 상태에서 분석만 실행:

```bash
# Step 0: 데이터 통합
python -m scripts.analyze_results --all

# Step 1: 행동 분석 (JobFair + LBOX) — [A]~[G] 통계 검정 + FDR + Cohen's d
python -m scripts.result_ans

# Step 2: SAE 분석 (JobFair + LBOX + Mind + WinoIdentity) — nFEP + H2 + feature overlap
python -m scripts.result_sae

# Step 3: WinoIdentity 분석 (pronoun→gender, cross-context, SAE 연동)
python -m scripts.result_wino
```

출력:
- `outputs/figures/` — 43개 플롯 PNG
- `outputs/figures/paper_table2.csv` — Paper Table 2 (FDR sig + Cohen's d)
- `outputs/figures/all_statistical_tests.json` — 전체 행동 통계
- `outputs/figures/sae_statistical_tests.json` — 전체 SAE 통계
- `outputs/figures/winoidentity_summary.json` — WinoIdentity 결과
- `outputs/figures/cross_3context.json` — 3-context 상관

---

## 실험 구조

| Context | Dataset | Task | Metric | Scale |
|---------|---------|------|--------|-------|
| **Coreference** | WinoIdentity (79,200) | 대명사 해석 A/B | Accuracy | 0-1 |
| **Job Application** | JobFair (15,900) | 이력서 0-10 점수 | Score | 0-10 (높을수록 유리) |
| **Legal Judgement** | LBOX (15,900) | 양형 0-10 점수 | Score | 0-10 (높을수록 엄격) |

Identity 구성: Baseline(1) + Single Gender(2) + Multi(2 gender × 25 traits = 50) = **53조건/문항**

### 방향 통일

```
JobFair: deviation = score(identity) − score(neutral)            [양수 = 유리]
LBOX:    deviation = −(score(identity) − score(neutral))         [양수 = 유리 (관대)]
Mind:    deviation = −(score(identity) − score(neutral))         [양수 = 유리 (건강)]
Wino:    deviation = accuracy(identity) − mean_accuracy          [양수 = 더 정확]
```

LBOX에서 높은 점수 = 엄격한 형벌 = 피고인에게 불리. Cross-context 비교 시 부호를 뒤집어 "양수 = favorable" 통일.

### 모델 (6개)

| Model | Provider | Type | SAE | Contexts |
|-------|----------|------|-----|----------|
| claude-sonnet-4.6 | Anthropic | API (safety) | — | JobFair, LBOX, Wino |
| claude-haiku-4.5 | Anthropic | API (safety) | — | JobFair, LBOX, Wino |
| gpt-4.1-mini | OpenAI | API | — | JobFair, LBOX, Wino |
| gemma-2-9b | Google (open) | Open-source | ✓ GemmaScope | JobFair, LBOX, Wino |
| gemma-3-4b | Google (open) | Open-source | — | JobFair, LBOX, Wino |
| mistral-7b | Mistral (open) | Open-source | — | JobFair only |

---

## 코드 구조

```
scripts/
│
│  ── 데이터 준비 ──
├── prepare_jobfair.py          ← JobFair CSV → WinoIdentity 형식 변환
├── prepare_lbox.py             ← LBOX CSV → WinoIdentity 형식 변환
│
│  ── 실험 실행 ──
├── run_one_batch.py            ← API 모델 Batch (OpenAI, Anthropic)
├── run_one_experiment.py       ← 오픈소스 모델 로컬 추론 (GPU)
├── run_sae_analysis.py         ← SAE feature 추출 + nFEP 계산 (GPU)
│
│  ── 결과 분석 (★ 핵심) ──
├── analyze_results.py          ← 결과 수집 → summary.json + consolidated CSV
├── result_ans.py               ← [A]~[G] 행동 통계 검정 + FDR + Cohen's d + bootstrap CI
├── result_sae.py               ← SAE nFEP 분석 + H2 상관 + feature overlap
├── result_wino.py              ← WinoIdentity 분석 + cross-context + SAE 연동
│
│  ── WinoIdentity Batch (coreference) ──
├── run_experiment.py           ← WinoIdentity 실험 구조 (batch/run_batch에서 호출)
├── run_batch.py                ← WinoIdentity coreference Batch 처리
│
│  ── 보조 ──
├── download_openai_results.py  ← OpenAI batch 결과 재다운로드
├── export_csv.py               ← (legacy)
├── visualize.py                ← (legacy, result_*.py에 통합됨)
├── run_behavioral.py           ← (legacy)
└── run_sae.py                  ← (legacy)
```

### 핵심 분석 파이프라인 (4단계)

```
analyze_results.py ──→ result_ans.py ──→ result_sae.py ──→ result_wino.py
      │                      │                  │                  │
      ▼                      ▼                  ▼                  ▼
 summary.json          [A]-[G] 검정       nFEP + H2          WinoIdentity
 consolidated.csv      FDR, Cohen's d     bootstrap CI        cross-context
                       paper_table2.csv   feature overlap     SAE linkage
```

---

## Metric 정의

| Metric | Formula | Range | Meaning |
|--------|---------|-------|---------|
| **Deviation** | `score(id) − score(neutral)` | ±10 | 양수=유리, 음수=불리 (LBOX는 부호 반전) |
| **Cohen's d** | `(μ₁ − μ₂) / s_pooled` | ±∞ | 효과 크기. 0.2=소, 0.5=중, 0.8=대 |
| **FDR q** | `p × n / rank` (BH method) | 0-1 | 다중비교 보정. q<0.05이면 유의 |
| **Pearson r** | `Σ(x−x̄)(y−ȳ) / √(...)` | -1 to +1 | 선형 상관. [CI]=bootstrap 95% |
| **Spearman ρ** | `1 − 6Σd²/(n(n²−1))` | -1 to +1 | 순위 상관. 이상치에 강건 |
| **Delta nFEP** | `‖Δf_inter − α·Δf_a − α·Δf_b‖ / (‖Δf_a‖+‖Δf_b‖)` | 0-1+ | SAE 합성 실패. 높을수록 collapse |
| **I_term** | `score(g+a) − score(g) − score(a) + score(base)` | ±∞ | 음수=교차 페널티, 0=가산적 |
| **Reversal rate** | `count(sign(job) ≠ sign(lbox)) / total` | 0-100% | 두 맥락에서 편향 방향이 반전되는 비율 |

### nFEP 상세 (본 연구의 기여)

```
기존 FEP (Khan et al. 2025):
  FEP = ‖f_inter − α₁·f_a − α₂·f_b‖

본 연구 Delta nFEP (3가지 확장):
  1. Baseline subtraction: Δf = f − f_baseline
  2. Normalization:        / (‖Δf_a‖ + ‖Δf_b‖)
  3. Multi-context/layer:  3 contexts × 3 layers = 9 analyses

  Delta nFEP = ‖Δf_inter − α₁·Δf_a − α₂·Δf_b‖ / (‖Δf_a‖ + ‖Δf_b‖)

Raw nFEP는 R²>0.98 (프롬프트 공유 내용이 지배) → Delta는 7-13× 크고 trait 차이가 뚜렷.
```

---

## 출력 구조

```
outputs/
├── analysis/                       ← analyze_results.py
│   ├── jobfair/
│   │   ├── consolidated_jobfair.csv   (15,900행 × 6 model 점수)
│   │   └── summary.json               (per-identity mean, std, n)
│   └── lbox/
│       ├── consolidated_lbox.csv
│       └── summary.json
│
├── behavioral/coref/               ← run_experiment / run_one_experiment
│   ├── raw_coref_gemma-2-9b.json      (79,200 records)
│   ├── raw_coref_gemma-3-4b.json
│   ├── metrics_coref_gemma-2-9b.json  (per-identity accuracy)
│   └── metrics_coref_gemma-3-4b.json
│
├── batch/                          ← API batch 결과
│   ├── openai/results_coref_gpt-4.1-mini_chunk*.jsonl
│   └── anthropic/results_coref_claude-*.jsonl
│
├── local/                          ← run_one_experiment
│   ├── jobfair/raw_jobfair_{model}.json
│   └── lbox/raw_lbox_{model}.json
│
├── sae/                            ← run_sae_analysis
│   ├── jobfair_gemma-2-9b_L{9,20,31}/sae_results.json  (150 groups)
│   ├── lbox_gemma-2-9b_L{9,20,31}/sae_results.json     (150 groups)
│   └── winoidentity_gemma-2-9b_L{9,20,31}/sae_results.json (2,000 groups)
│
└── figures/                        ← result_*.py (43 plots + 6 data files)
    ├── cross_context_{model}.png          (5)  [D] JobFair ↔ LBOX scatter
    ├── normalized_heatmap_{domain}.png    (2)  [E] trait × model z-deviation
    ├── wino_traits_{model}.png            (5)  Wino accuracy by trait
    ├── wino_heatmap_all_models.png        (1)  Wino all-model heatmap
    ├── wino_model_comparison.png          (1)  Wino accuracy + disparity bars
    ├── cross3_{model}_{ds}.png            (10) 3-Context scatter
    ├── nfep_traits_{ds}_gemma-2-9b_L{N}.png  (9) SAE trait nFEP
    ├── nfep_layers_{ds}_gemma-2-9b.png    (3)  SAE layer progression
    ├── h2_identity_{ds}_gemma-2-9b_L31.png   (2) H2 scoring
    ├── h2_nfep_vs_behavioral_{ds}_*.png   (2)  H2 trait-level
    ├── h2_wino_gemma-2-9b_L{N}.png       (3)  H2 WinoIdentity
    ├── paper_table2.csv                        Summary with FDR + Cohen's d
    ├── all_statistical_tests.json              Full behavioral stats
    ├── sae_statistical_tests.json              Full SAE stats
    ├── winoidentity_summary.json               Wino per-model results
    └── cross_3context.json                     3-context correlations
```

---

## 전체 실행 가이드 (처음부터)

### Step 0: 환경

```bash
pip install openai anthropic transformers accelerate sae-lens \
  huggingface_hub numpy scipy matplotlib statsmodels pandas
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

### Step 1: API 모델 실험 (Batch)

```bash
# JobFair + LBOX
for ds in jobfair lbox; do
  for prov in openai anthropic; do
    python -m scripts.run_one_batch prepare --data-file data/${ds}.csv --provider $prov
    python -m scripts.run_one_batch submit --provider $prov
  done
done

# WinoIdentity (별도 batch 구조)
for prov in openai anthropic; do
  python -m scripts.run_batch prepare --context coref --data-file data/winoidentity.csv --provider $prov
  python -m scripts.run_batch submit --provider $prov
done
```

### Step 2: 오픈소스 모델 (GPU 필요)

```bash
for ds in jobfair lbox; do
  for model in gemma-2-9b gemma-3-4b mistral-8b; do
    python -m scripts.run_one_experiment --data-file data/${ds}.csv --model $model
  done
done
```

### Step 3: SAE 분석 (GPU, ~1-2h per context)

```bash
for ds in data/jobfair.csv data/lbox.csv data/winoidentity.csv; do
  python -m scripts.run_sae_analysis --data-file $ds --model gemma-2-9b --multi-layer
done
```

### Step 5: 결과 분석 (CPU, ~5min)

```bash
python -m scripts.analyze_results --all
python -m scripts.result_ans
python -m scripts.result_sae
python -m scripts.result_wino
```

---

## 전체 데이터 접근

```python
import json, pandas as pd

# 전체 52 identity × 6 model 점수
df = pd.read_csv('outputs/analysis/jobfair/consolidated_jobfair.csv')
df.groupby('demographic')['score_gemma-2-9b'].mean().sort_values()

# 전체 77 identity accuracy (WinoIdentity)
with open('outputs/figures/winoidentity_summary.json') as f:
    wino = json.load(f)

# 전체 150 group nFEP (SAE)
with open('outputs/sae/jobfair_gemma-2-9b_L31/sae_results.json') as f:
    sae = json.load(f)['results']
for r in sorted(sae, key=lambda x: x['delta_nfep'], reverse=True):
    print(f"{r['combined']:30s} delta={r['delta_nfep']:.4f}")

# 전체 통계 검정 (FDR, Cohen's d, bootstrap CI 포함)
with open('outputs/figures/all_statistical_tests.json') as f:
    stats = json.load(f)
```

---

## Batch 관리

```bash
python -m scripts.run_one_batch poll --provider openai      # 진행 상황
python -m scripts.run_one_batch cancel --provider openai    # 전체 취소
python -m scripts.run_one_batch clean --provider openai     # 파일 정리
```

## API 호출 수

```
JobFair:      15,900 × 1 = 15,900 calls/model
LBOX:         15,900 × 1 = 15,900 calls/model
Mind:         15,900 × 1 = 15,900 calls/model
WinoIdentity: 79,200 × 1 = 79,200 calls/model

Batch chunk sizes: OpenAI=1,000, Anthropic=5,000
```