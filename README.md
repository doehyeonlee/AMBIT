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

추가 지원(행동 측정 `run_one_experiment` · 제거 `run_removal`): gemma-3-12b, gemma-3-27b, llama-3.1-8b, qwen-3-8b, qwen-3.5-9b, qwen-3.5-27b

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
│  ── 3부: 편향 제거 + 압축 통제 판정 ──
├── removal_models.py           ← 모델·SAE 레지스트리 (Gemma 2/3, Qwen3/3.5, Llama 3.1 …), 층 자동 탐지
├── removal_methods.py          ← 제거 기법 6종, 개입 hook, 0~10 확률 채점, 판정(gate), 프로브
├── run_removal.py              ← 맥락 간 전이 행렬 실행기 (중단 후 이어 돌기)
├── run_parallel.py             ← 여러 모델을 GPU마다 하나씩 동시에 실행 (behavior / sae / removal)
├── setup_env.sh                ← 환경 점검: 패키지, torch↔torchvision, GPU, HF 로그인
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
bash scripts/setup_env.sh      # 처음 한 번: 패키지 설치, torch↔torchvision 버전 맞추기, GPU·HF 로그인 점검

# GPU 여러 장에 모델을 하나씩 동시에 (빈 GPU가 생기면 다음 작업 자동 시작, 끝난 작업은 건너뜀)
python -m scripts.run_parallel behavior --models gemma-2-9b,gemma-3-4b,mistral-7b --datasets jobfair,lbox,mind

# 한 개씩 돌릴 때
python -m scripts.run_one_experiment --data-file data/jobfair.csv --model gemma-2-9b
```

- 1,000행마다 `outputs/local/<ds>/raw_<ds>_<model>.partial.json`에 중간 저장하고, 다시 실행하면 이어서 돈다.
- `RuntimeError: operator torchvision::nms does not exist` → torch와 torchvision 버전 불일치. `bash scripts/setup_env.sh`가 고친다.
- 27B 모델은 40GB GPU 2장을 묶어 쓴다 (80GB GPU면 1장).

### 여러 모델 동시 실행 (`scripts/run_parallel.py`)

| 작업 | 단위 | 예시 |
|---|---|---|
| `behavior` | 모델 × 데이터셋 | `python -m scripts.run_parallel behavior --models gemma-3-12b,qwen-3-8b --datasets jobfair,lbox,mind` |
| `sae` | 모델 × 데이터셋 (층 전부) | `python -m scripts.run_parallel sae --models gemma-2-9b,gemma-3-12b,qwen-3-8b --datasets lbox,mind` |
| `removal` | 모델 (세 맥락 한 번에) | `python -m scripts.run_parallel removal --models gemma-3-12b,qwen-3-8b,llama-3.1-8b,gemma-2-9b --extra "--methods pca,leace,diffmean,affine,sae"` |

`--gpus 0,1`로 쓸 GPU를 고르고, `--dry-run`으로 배치 계획만 볼 수 있다. 작업별 로그는 `logs/<task>_<model>_<ds>.log`, 요약은 `logs/parallel_<task>.json`.

### Step 3: SAE 분석 (GPU, ~1-2h per context)

```bash
for ds in data/jobfair.csv data/lbox.csv data/winoidentity.csv; do
  python -m scripts.run_sae_analysis --data-file $ds --model gemma-2-9b --multi-layer
done
```

> **2026-10 수정 — 해당 결과는 다시 돌릴 것**
> - 이전에는 `max_length=512`로 **오른쪽**을 잘라, 512토큰이 넘는 프롬프트(LBOX 약 13%, Mind 약 33%)는 점수 지시문이 잘린 채 본문 중간 토큰의 특징을 읽었다. 이제 왼쪽을 자르고 한도를 4096으로 올려 마지막 토큰이 항상 프롬프트 끝이다. → **LBOX · Mind의 ΔnFEP 재실행 필요** (JobFair는 토큰 수 추정상 모두 512 이하라 영향 없음)
> - Llama Scope는 공개 hyperparams대로 JumpReLU(θ, 디코더 노름) + dataset-wise 입력 정규화로 인코딩한다(이전: TopK). → **llama-3.1-8b SAE 결과 재실행 필요**
> - 층 접근을 `model.model.layers` 하드코딩에서 아키텍처 자동 탐지로 바꿔 Gemma 3 멀티모달 래퍼에서도 동작한다.
> - 추가 모델: `gemma-3-12b`(Gemma Scope 2, 층 12/24/31/41), `qwen-3-8b`(Qwen-Scope, 층 8/18/27). `gemma-3-4b`는 HF 직접 로드로 변경.
>
> ```bash
> for ds in data/lbox.csv data/mind.csv; do python -m scripts.run_sae_analysis --data-file $ds --model gemma-2-9b --multi-layer; done
> for m in gemma-3-12b qwen-3-8b; do for ds in data/jobfair.csv data/lbox.csv data/mind.csv; do
>   python -m scripts.run_sae_analysis --data-file $ds --model $m --multi-layer; done; done
> ```

### Step 5: 결과 분석 (CPU, ~5min)

```bash
python -m scripts.analyze_results --all
python -m scripts.result_ans
python -m scripts.result_sae
python -m scripts.result_wino
```

---

## 3부: 편향 제거 + 압축 통제 판정

격차가 줄었을 때 그것이 **정체성 제거**인지, 출력 점수가 좁아진 **분산 압축**인지 가린다.

```
ΔD  = disparity(before) − disparity(after)          격차 감소
C   = 1 − Var(score_after) / Var(score_before)       출력 압축률
ΔD_ctrl(C)  ← 정체성과 무관한 대조군(무작위 방향 · 전체 활성 PCA)을 여러 강도로 돌려 보간
E   = ΔD − ΔD_ctrl(C)                                초과 감소 (문항 단위 부트스트랩 95% CI)
행동 개선: CI 하한 > 0      표현 제거: MLP 프로브 ≤ 우연 + 0.05
→ 진짜 제거 / 분리(행동만) / 비활성 소거(표현만) / 압축·무효
```

### 제거 기법 (추론 시점, 잔차 스트림 hook)

| 기법 | 개입 | 방향 추정 | 강도 |
|---|---|---|---|
| `pca` | h′ = h − α·UUᵀh | 정체성별 평균 − 중립 평균의 상위 k 주성분 | α |
| `leace` | h′ = h − α·W⁺PW(h − μ) | 활성값 + 정체성 원-핫 (닫힌 해) | α |
| `diffmean` | h′ = h − α(h·v̂)v̂ | v = μ정체성 − μ중립 | α |
| `affine` | h′ = h − α((h − μ중립)·v̂)v̂ | 정체성 / 중립 평균 | α |
| `steer` | h′ = h − α·v | 대조 평균 차이 | α |
| `sae` | h′ = h − α·Σᵢ∈S fᵢ(h)dᵢ (재구성 오차 유지) | 중립 대비 활성 차이 상위 특징 | 특징 수, α |

층은 **깊이 비율**(`--band 0.43,0.62`, Gemma-2-9B의 18~26층)로 지정하고, 개입하는 층마다 그 층의 활성값으로 방향을 따로 추정한다. SAE 소거는 SAE가 있는 층 하나에만 적용한다.

### 지원 모델 (`python -m scripts.run_removal --list-models`)

| key | 공개 SAE | SAE 학습 | 용도 | GPU |
|---|---|---|---|---|
| `gemma-3-12b` | Gemma Scope 2 | IT | **주력**, 5부 이미지 입력과 연결 | 48GB |
| `qwen-3-8b` | Qwen-Scope | Base | 4부 사고 모드와 공유 | 24~48GB |
| `llama-3.1-8b` | Llama Scope | Base | 기존 연구(Karvonen & Marks 등) 비교 | 24~48GB |
| `gemma-2-9b` | Gemma Scope (IT) | IT | 2부 ΔnFEP · 예비 감사 재현 | 48GB |
| `gemma-3-4b` | Gemma Scope 2 | IT | 개발 · 디버깅 | 24GB |
| `qwen-3.5-9b`, `qwen-3.5-27b` | Qwen-Scope | Base / IT | 최신 모델, 규모 확장 | 24GB / 80GB |
| `mistral-7b`, `qwen-2.5-7b`, `r1-distill-llama-8b` | — | — | SAE 제외 기법만 | 24GB |

새 모델은 `REMOVAL_MODELS`에 한 줄 추가하면 된다. Base로 학습된 SAE는 실행 로그의 `재구성 EV`를 확인하고, 0.7 미만이면 SAE 결과는 해석하지 않는다.

### 실행

```bash
pip install -r requirements.txt
python -m tests.test_removal_tiny                     # GPU 없이 파이프라인 점검 (작은 무작위 모델)

# 10분 점검
python -m scripts.run_removal --model gemma-3-4b --n-train 4 --n-test 4 --methods diffmean --alphas 1.0

# 본 실행 (중단돼도 같은 명령으로 이어서 돈다)
nohup python -m scripts.run_removal --model gemma-3-12b \
    --methods pca,leace,diffmean,affine,sae --alphas 0.5,1.0,2.0 > removal_g3.log 2>&1 &

# 오늘 밤 축소판: 정체성 10개
python -m scripts.run_removal --model gemma-2-9b --methods pca,leace,diffmean \
    --identities male,female,female+black,male+black,female+old,male+rich,female+poor,male+gay,female+disabled,male+immigrant
```

결과: `outputs/removal/<model>.json` — `runs["diffmean|jobfair|1.0"]["eval"]["lbox"]`에 E, CI, 압축률, 프로브, 판정이 들어 있다.

### 서버 (2026년 10월 시세, 시간당)

| GPU | 가격 | 돌릴 수 있는 모델 |
|---|---|---|
| RTX A6000 / L40 (48GB) | $0.35~0.82 | 9~12B 전부 — **기본 추천** |
| RTX 4090 (24GB) | ≈ $0.5 | 8B 이하 |
| A100 80GB | $1.09~1.79 | 27B |

기본 설정(맥락당 평가 30문항 × 53행 × 3맥락 ≈ 4,800행/조건, 조건 ≈ 45개)으로 12B 모델 하나가 48GB GPU에서 약 10시간(추정). 모델 캐시는 영구 볼륨에 두고(`HF_HOME`), 스팟 인스턴스는 이어 돌기와 함께 쓴다. Gemma·Llama는 Hugging Face에서 라이선스 동의 후 `hf auth login`.

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