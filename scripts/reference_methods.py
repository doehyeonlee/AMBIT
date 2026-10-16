"""
3부: 원 논문의 공개 코드를 그대로 따르는 비교 기법

기존 removal_methods.py의 기법(pca, leace, diffmean, affine, steer, sae)은 AMBIT 설정에 맞춘 구현이다.
여기 기법들은 각 논문이 공개한 코드의 절차를 그대로 옮겨, "원 논문 방식으로 돌려도 같은 판정이 나오는가"를 본다.
판정(압축 통제 E, 프로브)은 기존 기법과 똑같이 run_removal.py가 적용한다.

  키           원 논문 · 공개 코드                                      라이선스   옮긴 방식
  leace_ref    Belrose et al. 2023 · github.com/EleutherAI/concept-erasure   MIT      패키지를 직접 호출 (LeaceFitter)
  km_ace       Karvonen & Marks 2025 · github.com/adamkarvonen/llm_bias      없음     절차를 재구현 (코드 복사 안 함)
  arditi       Arditi et al. 2024 · github.com/andyrdt/refusal_direction     Apache   절차를 재구현
  caa          Rimsky(Panickssery) et al. 2024 · github.com/nrimsky/CAA      MIT      절차를 재구현
  splice       Holstege, Ravfogel, Wouters (NeurIPS 2025) · opt-separation    없음     절차를 재구현 (proj.opt_sep_proj)

각 기법의 원 코드 대비 차이는 함수 docstring의 "원 코드와 다른 점"에 적었다.
라이선스가 없는 저장소(llm_bias)는 코드를 옮기지 않고 같은 계산을 새로 작성했으며,
tests/test_reference_methods.py가 (원 저장소를 내려받은 경우) 원 함수와 수치가 같은지 확인한다.

α: 원 논문들은 완전 적용(α = 1)만 쓴다. AMBIT은 압축 수준을 바꿔 가며 판정하려고 α를 곱한다.
   α = 1일 때 각 편집은 원 코드의 출력과 같다.
"""
from __future__ import annotations
import numpy as np, torch
from collections import defaultdict
from scripts.removal_models import get_layers
from scripts.removal_methods import _encode, _device, by_identity

GENDERS = ("male", "female")


# ---------------------------------------------------------------- 공통: 토큰 평균 활성값 (Karvonen & Marks 방식)
def collect_acts_mean(model, tok, items, layers, batch=8):
    """층별 프롬프트 전체 토큰 평균 활성값 {layer: (N, d)}.
    llm_bias의 compute_mean_activations_per_prompt와 같다: 패딩을 0으로 만들고 합을 실제 길이로 나눈다."""
    layers = list(layers)
    blocks, store, mask = get_layers(model), {l: [] for l in layers}, {}
    def mk(l):
        def hook(mod, inp, out):
            hs = (out[0] if isinstance(out, tuple) else out).float()
            m = mask["m"]
            store[l].append(((hs * m[..., None]).sum(1) / m.sum(1, keepdim=True)).cpu())
        return hook
    hs = [blocks[l].register_forward_hook(mk(l)) for l in layers]
    try:
        for i in range(0, len(items), batch):
            enc = _encode(tok, [it["text"] for it in items[i:i + batch]], _device(model))
            mask["m"] = enc["attention_mask"].float()
            model(**enc, use_cache=False, logits_to_keep=1)
    finally:
        for h in hs:
            h.remove()
    return {l: torch.cat(v).numpy() for l, v in store.items()}


# ---------------------------------------------------------------- 1) LEACE: concept-erasure 패키지 그대로
class RefLeaceEdit:
    """concept_erasure.LeaceEraser를 α배로 적용. α = 1이면 eraser(h)와 같다.
    LeaceEraser.__call__:  x − ((x − bias) @ proj_rightᴴ) @ proj_leftᴴ"""
    def __init__(self, eraser):
        self.L = eraser.proj_left.detach().cpu().numpy()
        self.R = eraser.proj_right.detach().cpu().numpy()
        self.b = None if eraser.bias is None else eraser.bias.detach().cpu().numpy()

    def apply(self, h, alpha):
        L = torch.as_tensor(self.L, dtype=torch.float32, device=h.device)
        R = torch.as_tensor(self.R, dtype=torch.float32, device=h.device)
        x = h.float()
        d = x - torch.as_tensor(self.b, dtype=torch.float32, device=h.device) if self.b is not None else x
        return (x - alpha * (d @ R.T) @ L.T).to(h.dtype)


def fit_leace_ref(acts, items, **fitter_kw):
    """Belrose et al. 공식 구현(LeaceFitter, 기본 설정: 축소 공분산 · svd_tol 0.01 · 공분산 대각합 제약 · affine).
    z = 정체성 원-핫(중립 = 0 벡터). 기존 fit_leace(닫힌 해 + 리지)와 달리 표본 < 차원일 때도 수치적으로 안정적이다.
    원 코드와 다른 점: 원 논문의 concept scrubbing은 앞 층을 지운 뒤 다음 층을 추정하지만, 여기서는 다른 기법과
    같게 층마다 원래 활성값으로 추정한다."""
    from concept_erasure import LeaceFitter
    ids = sorted({it["identity"] for it in items} - {"neutral"})
    Z = torch.tensor([[it["identity"] == c for c in ids] for it in items], dtype=torch.float64)
    X = torch.as_tensor(np.asarray(acts), dtype=torch.float64)
    return RefLeaceEdit(LeaceFitter.fit(X, Z, **fitter_kw).eraser)


# ---------------------------------------------------------------- 2) Karvonen & Marks: 백화 방향 + 중점 아핀 편집
def km_bias_types(items, types="all"):
    """llm_bias의 bias type(인종 · 성별처럼 이진 속성)을 AMBIT 정체성에 대응시킨다. 원 코드처럼 양·음 표본 수가 같다.
      gender     : female이 들어간 행 vs male이 들어간 행 (문항당 26 vs 26)
      <속성 a>   : male+a · female+a  vs  male · female      (문항당 2 vs 2, 성별을 맞춘 대조)
    types="gender"면 성별 방향 하나만."""
    ident = [it["identity"] for it in items]
    parts = [set(s.split("+")) if s != "neutral" else set() for s in ident]
    out = {"gender": (np.array(["female" in p for p in parts]), np.array(["male" in p for p in parts]))}
    if types == "all":
        attrs = sorted(set().union(*parts) - set(GENDERS))
        gender_only = np.array([len(p) == 1 and next(iter(p)) in GENDERS for p in parts])
        for a in attrs:
            pos = np.array([a in p for p in parts])
            if pos.sum() and pos.sum() == gender_only.sum():
                out[a] = (pos, gender_only)
    return out


class KMAceEdit:
    """llm_bias get_projection_ablation_hook:  h ← h − Σ_k ((h · u_k) − b_k) u_k   (float32, 직교화 없이 합)"""
    def __init__(self, U, b):
        self.U, self.b = U.astype(np.float32), b.astype(np.float32)   # (K, d), (K,)

    def apply(self, h, alpha):
        U = torch.as_tensor(self.U, device=h.device)
        b = torch.as_tensor(self.b, device=h.device)
        proj = h.float() @ U.T - b
        return (h.float() - alpha * proj @ U).to(h.dtype)


def fit_km_ace(acts, items, types="all", eps=1e-4):
    """llm_bias get_mean_diff_activations와 같은 계산:
      u = (μ₊ − μ₋) / sqrt(Var[양·음 합친 표본] + 1e-4),  u ← u/‖u‖
      b = ½(mean(x₊·u) + mean(x₋·u))     (두 집단 투영 평균의 중점)
    acts는 토큰 평균 활성값(collect_acts_mean)이어야 원 코드와 같다.
    원 코드와 다른 점: 방향 추정 데이터가 Tamkin et al. 합성 데이터가 아니라 AMBIT 학습 문항이고,
    인종 · 성별 대신 km_bias_types의 속성 방향을 쓴다. 개입 층은 원 코드 기본값(전 층)을 run_removal에서 따른다."""
    X = np.asarray(acts, dtype=np.float32)
    U, B = [], []
    for name, (pos, neg) in km_bias_types(items, types).items():
        xp, xn = X[pos], X[neg]
        var = np.concatenate([xp, xn]).var(0) + eps
        u = (xp.mean(0) - xn.mean(0)) / np.sqrt(var)
        u = u / np.linalg.norm(u)
        U.append(u); B.append(0.5 * ((xp @ u).mean() + (xn @ u).mean()))
    return KMAceEdit(np.stack(U), np.array(B))


# ---------------------------------------------------------------- 3) Arditi et al.: 한 방향을 모든 층에서 소거
def _write_modules(block):
    """블록이 잔차 스트림에 더하는 두 출력. Gemma 2/3는 샌드위치 정규화라 attn/mlp 뒤 정규화 출력이 실제로 더해진다."""
    if hasattr(block, "post_feedforward_layernorm") and hasattr(block, "post_attention_layernorm"):
        return [block.post_attention_layernorm, block.post_feedforward_layernorm]
    attn = getattr(block, "self_attn", None) or getattr(block, "attn", None)
    mlp = getattr(block, "mlp", None) or getattr(block, "feed_forward", None)
    return [m for m in (attn, mlp) if m is not None]


class ArditiAblation:
    """refusal_direction의 directional ablation:  x ← x − (x · r̂) r̂
    모든 층의 블록 입력(pre-hook)과 attn · mlp 출력(hook)에 적용해 잔차 스트림 전체에서 방향을 지운다.
    run_removal의 Intervention이 register()를 호출한다."""
    def __init__(self, r, layer):
        self.r = (r / np.linalg.norm(r)).astype(np.float32)
        self.layer = layer   # 방향을 뽑은 층 (기록용)

    def _abl(self, x, alpha):
        r = torch.as_tensor(self.r, device=x.device, dtype=x.dtype)
        return x - alpha * (x @ r)[..., None] * r

    def register(self, model, alpha):
        hs = []
        for blk in get_layers(model):
            def pre(mod, args, kwargs):
                if args:
                    return (self._abl(args[0], alpha),) + tuple(args[1:]), kwargs
                kwargs["hidden_states"] = self._abl(kwargs["hidden_states"], alpha)
                return args, kwargs
            hs.append(blk.register_forward_pre_hook(pre, with_kwargs=True))
            for wm in _write_modules(blk):
                def post(mod, inp, out):
                    if isinstance(out, tuple):
                        return (self._abl(out[0], alpha),) + tuple(out[1:])
                    return self._abl(out, alpha)
                hs.append(wm.register_forward_hook(post))
        return hs


def arditi_candidates(acts_by_layer, items):
    """층마다 r_l = mean(정체성) − mean(중립) (마지막 토큰). refusal_direction의 get_mean_diff와 같다."""
    out = {}
    for l, a in acts_by_layer.items():
        g = by_identity(a, items)
        r = np.concatenate([x for k, x in g.items() if k != "neutral"]).mean(0) - g["neutral"].mean(0)
        out[l] = ArditiAblation(r, l)
    return out


# ---------------------------------------------------------------- 4) CAA: 한 층에서 평균 차이 벡터를 지시문 끝 이후에 더함
class CAAEdit:
    """nrimsky/CAA add_vector_from_position: 지시문이 끝난 위치부터 h ← h + m·v (정규화하지 않은 평균 차이).
    채점 프롬프트에서는 지시문 끝 = 마지막 프롬프트 토큰이므로 마지막 위치(와 이어지는 KV 캐시 스텝)에만 적용한다.
    제거 방향이므로 m = −α."""
    def __init__(self, v, layer):
        self.v, self.layer = v.astype(np.float32), layer

    def apply(self, h, alpha):
        v = torch.as_tensor(self.v, device=h.device, dtype=h.dtype)
        h = h.clone()
        h[:, -1:] = h[:, -1:] - alpha * v
        return h


def caa_candidates(acts_by_layer, items):
    """층마다 v_l = 문항별 (정체성 − 같은 문항 중립) 차이의 평균 (마지막 토큰). CAA generate_vectors의 쌍 평균 차이."""
    out = {}
    for l, a in acts_by_layer.items():
        base = {(it["context"], it["item_id"]): x for x, it in zip(a, items) if it["identity"] == "neutral"}
        diffs = [x - base[(it["context"], it["item_id"])] for x, it in zip(a, items) if it["identity"] != "neutral"]
        out[l] = CAAEdit(np.mean(diffs, 0), l)
    return out


# ---------------------------------------------------------------- 5) SPLICE: 정체성 선형 보호 + 과제 공분산 보존
class LowRankEdit:
    """h' = h − α · L R (h − μ).  (I − P)가 낮은 계수일 때 d×d 행렬 대신 L(d×k), R(k×d)만 저장한다."""
    def __init__(self, L, R, mu):
        self.L, self.R, self.mu = L.astype(np.float32), R.astype(np.float32), mu.astype(np.float32)

    def apply(self, h, alpha):
        L = torch.as_tensor(self.L, device=h.device)
        R = torch.as_tensor(self.R, device=h.device)
        x = h.float() - torch.as_tensor(self.mu, device=h.device)
        return (h.float() - alpha * (x @ R.T) @ L.T).to(h.dtype)


def _cov(X, Y):
    """교차 공분산 (열 평균을 뺀 뒤 Xᵀ Y / n). 방향만 쓰이므로 정규화 상수는 결과에 영향이 없다."""
    Y = Y.reshape(len(Y), -1)
    return (X - X.mean(0)).T @ (Y - Y.mean(0)) / len(X)


def _orth_basis(A, tol=1e-10):
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, s > tol * max(s.max(), 1e-300)]


def fit_splice(acts, items, y, eps=1e-8):
    """SPLICE (Holstege, Ravfogel, Wouters 2025)의 opt_sep_proj와 같은 계산.
      백화 W = Cov(X)^(-1/2) (고윳값 < eps는 0으로) → v = Cov(XW, z), u = Cov(XW, y)
      백화 공간에서 v 방향을 지우되(kernel) u는 보존하는(range) 사영:  P' = I − V (ẼᵀV)⁻¹ Ẽᵀ,
        Ẽ = span{u, v} 안에서 u에 직교하는 정규직교 기저 (원 코드 find_orthogonal_vector_multi와 같은 부분공간)
      원 공간: h' = h − W⁺ V (ẼᵀV)⁻¹ Ẽᵀ W (h − μ)
    z = 정체성 지시 변수(중립 = 0 벡터), y = 문항별 중립 점수(정체성 없이 모델이 준 점수, 같은 문항의 모든 행에 같은 값).
    원 코드와 다른 점 하나: 원 코드는 P* = W⁺P'W라서 표본 수 < 차원이면 학습 데이터가 덮지 않는 방향을 모두 0으로
    만든다(우리 설정: 1,060행 < 3,584차원). 여기서는 I − W⁺(I − P')W로 써서 그 방향은 그대로 둔다(LEACE와 같은 처리).
    Cov(X)가 정칙이면 두 식은 정확히 같다 (tests/test_reference_methods.py)."""
    X = np.asarray(acts, dtype=np.float64)
    ids = sorted({it["identity"] for it in items} - {"neutral"})
    Z = np.array([[it["identity"] == c for c in ids] for it in items], float)
    mu = X.mean(0)
    Xc = X - mu
    ev, Vec = np.linalg.eigh(np.cov(Xc, rowvar=False))
    ev[ev < eps] = 0
    nz = ev > 0
    W = (Vec[:, nz] / np.sqrt(ev[nz])) @ Vec[:, nz].T
    Winv = (Vec[:, nz] * np.sqrt(ev[nz])) @ Vec[:, nz].T
    XW = Xc @ W
    V = _orth_basis(_cov(XW, Z))
    u = _cov(XW, np.asarray(y, float))
    u = u / np.linalg.norm(u)
    Q = _orth_basis(np.column_stack([u, V]))
    Et = _orth_basis(Q - u @ (u.T @ Q))[:, :V.shape[1]]
    L = Winv @ V @ np.linalg.pinv(Et.T @ V)
    return LowRankEdit(L, Et.T @ W, mu)
