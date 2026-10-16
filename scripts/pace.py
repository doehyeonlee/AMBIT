"""
PACE: Paired, Anchored, Compression-aware Erasure  (AMBIT 제안 기법, 초안)

감사 결과 세 가지를 설계 원칙으로 바꾼 기법이다.
  1) 격차 감소는 대부분 분산 압축이었다       → 과제 신호의 분산을 건드리지 않는 방향으로만 편집한다
  2) 정체성 정보 제거 ≠ 행동 변화 (J4 ↔ N1)   → 정체성 "정보"가 아니라 점수에 미치는 "효과"를 지운다
  3) 인과 구간(L23~26) · 맥락마다 다른 특징   → 인과 구간에서, 맥락별로 추정하고 맥락에 따라 골라 적용한다

수식 (층 l, 맥락 c의 학습 문항):
  Δ_{i,g} = h(i, g) − h(i, 중립)                      같은 문항의 짝지은 반사실 차이 (정체성 효과)
  S_Δ = E[Δ Δᵀ]                                        정체성 효과의 2차 모멘트
  S_T = Cov_i(h(i, 중립)) + ε·(tr S_T / d)·I            문항 간 과제 신호 (정답 라벨 불필요)
        ε는 축소 계수다. 표본이 적으면 분산이 작은 방향이 과장되어 PACE가 지나치게 보수적이 된다(장난감: ε 0.1 → 효과 62% 남음,
        ε 5 → 36%, 이론 하한 32%). 정체성 효과가 과제 방향과 겹치는 몫은 과제를 해치지 않고는 지울 수 없으므로 남는다 —
        이 남은 몫이 점수 격차에 실제로 영향을 주는지는 압축 통제 판정(E)이 가린다.
  S_Δ u = λ S_T u  의 상위 k개 → U                       과제 분산 단위당 정체성 효과가 가장 큰 방향
  h' = h − α · U (Uᵀ M U)⁻¹ Uᵀ M (h − μ_중립,c),  M = S_T⁻¹  (앵커 = 중립 평균, 읽기 잣대 = 과제 역공분산)

명제(초안): 범위가 span(U)인 선형 편집 h − U Bᵀ(h − μ) (BᵀU = I) 중 지우는 과제 분산 tr(Bᵀ S_T B · UᵀU)를
최소로 만드는 B는 B = M U (Uᵀ M U)⁻¹ (일반화 최소제곱과 같은 구조). tests/test_pace.py가 수치로 확인한다.

비교군과 구분되는 점 (README · 보고서 11장과 같은 내용):
  - LEACE / leace_ref : 정체성의 "선형 복원 가능성"을 지움, 과제 신호를 따로 지키지 않음
  - SPLICE            : 선형 복원 가능성을 지우면서 과제 라벨 y와의 "공분산"을 보존 (y 필요)
  - ACE · km_ace      : 평균 차이 방향 하나를 중립/중점 앵커로 이동, 과제 신호 고려 없음
  - CAA · steer       : 평균 차이 벡터를 빼기, 중립 입력도 같이 밀림
  - PACE              : 문항별 짝으로 잰 정체성 "효과"를, 문항 간 과제 "분산"(라벨 없이)을 최소로 건드리게 지움.
                        정체성이 여전히 디코딩될 수 있다(판정 = 분리). 이것이 PACE의 주장이다:
                        공정한 결정에 필요한 것은 정보의 삭제가 아니라 효과의 차단이다.

절제(ablation) 키 (run_removal --methods):
  pace           기본: 짝 Δ, M = S_T⁻¹, 중립 앵커, 인과 구간(--pace-band), 학습 맥락 하나로 추정 (전이 행렬에 들어감)
  pace_routed    맥락마다 추정하고 추론 때 맥락을 골라 적용 (학습 맥락 키 = "all")
  pace_eye       M = I (일반 직교 사영) → 명제의 효과 확인
  pace_group     짝 대신 집단 평균 차이 → 짝지은 반사실의 효과 확인
  pace_noanchor  앵커 없이 0으로 → 앵커의 효과 확인
  pace_early     인과 구간 대신 공통 개입 구간(--band) → 개입 위치의 효과 확인
"""
from __future__ import annotations
import numpy as np, torch
from scripts.removal_methods import by_identity


class PaceEdit:
    """h' = h − α · L R (h − μ),  L = U (UᵀMU)⁻¹ (d×k),  R = UᵀM (k×d)"""
    def __init__(self, L, R, mu, info=None):
        self.L, self.R, self.mu = L.astype(np.float32), R.astype(np.float32), mu.astype(np.float32)
        self.info = info or {}

    def delta(self, h):
        L = torch.as_tensor(self.L, device=h.device)
        R = torch.as_tensor(self.R, device=h.device)
        x = h.float() - torch.as_tensor(self.mu, device=h.device)
        return (x @ R.T) @ L.T

    def apply(self, h, alpha):
        return (h.float() - alpha * self.delta(h)).to(h.dtype)


def paired_effects(acts, items):
    """Δ = h(i, g) − h(i, 중립), 같은 (context, item_id)끼리."""
    base = {(it["context"], it["item_id"]): a for a, it in zip(acts, items) if it["identity"] == "neutral"}
    return np.array([a - base[(it["context"], it["item_id"])] for a, it in zip(acts, items) if it["identity"] != "neutral"])


def fit_pace(acts, items, neutral_extra=None, k=4, eps=1.0, metric="task", delta="paired", anchor=True):
    """acts: 학습 문항 활성값 (N, d), items: 같은 순서의 문항 dict.
    neutral_extra: 학습·평가 문항이 아닌 문항들의 중립 활성값 (S_T 추정을 안정시키는 용도, 평가 문항과 겹치지 않음)."""
    from scipy.linalg import eigh
    X = np.asarray(acts, dtype=np.float64)
    g = by_identity(X, items)
    neu = g["neutral"]
    if delta == "paired":
        D = paired_effects(X, items)
    else:   # group: 정체성별 평균 − 중립 평균
        D = np.stack([v.mean(0) - neu.mean(0) for kk, v in g.items() if kk != "neutral"])
    S_D = D.T @ D / len(D)
    T = neu if neutral_extra is None else np.concatenate([neu, np.asarray(neutral_extra, dtype=np.float64)])
    Tc = T - T.mean(0)
    S_T = Tc.T @ Tc / max(len(T) - 1, 1)
    d = X.shape[1]
    S_T = S_T + eps * (np.trace(S_T) / d) * np.eye(d)
    w, U = eigh(S_D, S_T, subset_by_index=[d - k, d - 1])
    U = U[:, ::-1]
    if metric == "task":
        MU = np.linalg.solve(S_T, U)              # M U,  M = S_T⁻¹
    else:
        U, _ = np.linalg.qr(U)
        MU = U
    L = U @ np.linalg.inv(U.T @ MU)
    mu = neu.mean(0) if anchor else np.zeros(d)
    # 기록: 지우는 정체성 효과 비율과 과제 분산 비율 (평가에는 쓰지 않음)
    Pd = D @ MU @ L.T
    PT = Tc @ MU @ L.T
    info = dict(k=k, gen_eig=w[::-1].tolist(), n_task=len(T),
                effect_removed=float((Pd ** 2).sum() / (D ** 2).sum()),
                task_var_removed=float((PT ** 2).sum() / (Tc ** 2).sum()))
    return PaceEdit(L, MU.T, mu, info)


class PaceRouted:
    """맥락별 PACE 편집을 두고, 시퀀스마다 마지막 토큰 활성값이 가장 가까운 맥락 중심(코사인)의 편집을 적용한다.
    KV 캐시 1스텝(길이 1)에서는 직전 프롬프트에서 고른 맥락을 그대로 쓴다."""
    def __init__(self, edits: dict, centroids: dict):
        self.names = list(edits)
        self.edits = [edits[c] for c in self.names]
        C = np.stack([centroids[c] for c in self.names])
        self.C = (C / np.linalg.norm(C, axis=1, keepdims=True)).astype(np.float32)
        self._last = None

    def route(self, h):
        if h.shape[1] == 1 and self._last is not None and len(self._last) == h.shape[0]:
            return self._last
        x = h[:, -1].float()
        x = x / x.norm(dim=-1, keepdim=True)
        self._last = (x @ torch.as_tensor(self.C, device=h.device).T).argmax(-1)
        return self._last

    def apply(self, h, alpha):
        r = self.route(h)
        out = h.float().clone()
        for j, e in enumerate(self.edits):
            m = r == j
            if m.any():
                out[m] = out[m] - alpha * e.delta(h[m])
        return out.to(h.dtype)


def fit_pace_routed(acts_by_ctx: dict, items_by_ctx: dict, neutral_extra_by_ctx: dict | None = None, **kw):
    edits, cents = {}, {}
    for c, a in acts_by_ctx.items():
        ne = (neutral_extra_by_ctx or {}).get(c)
        edits[c] = fit_pace(a, items_by_ctx[c], ne, **kw)
        cents[c] = np.asarray(a).mean(0)
    return PaceRouted(edits, cents)
