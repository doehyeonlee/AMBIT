"""
제안 기법 PACE(scripts/pace.py) 점검 (CPU, 몇 초)
  python -m tests.test_pace

1) 명제: 같은 방향 U를 지우는 선형 편집 중 M = S_T⁻¹로 읽는 PACE가 지우는 과제 분산이 가장 작다
   (M = I 직교 사영, 무작위로 고른 다른 유효한 읽기 B와 비교)
2) 앵커: 중립 평균 활성값은 편집 후에도 그대로
3) 효과 차단: α = 1에서 짝지은 정체성 효과의 U 좌표가 0
4) 장난감 비교: 정체성 효과가 과제 방향과 일부 겹칠 때, 같은 양의 효과를 지우면서 DiffMean · 집단 평균 변형보다 과제 분산을 덜 지움
5) 라우팅: 맥락 중심이 분리될 때 시퀀스마다 맞는 맥락의 편집을 고름
"""
import numpy as np, torch
from scripts import pace as P, removal_methods as R

rng = np.random.default_rng(0)


def toy(n_items=40, d=48, overlap=0.6, ctx="jobfair"):
    """과제 신호(문항 품질 q)는 방향 t, 정체성 효과는 방향 e = overlap·t + √(1−overlap²)·t⊥ 를 따른다."""
    t = rng.standard_normal(d); t /= np.linalg.norm(t)
    tp = rng.standard_normal(d); tp -= (tp @ t) * t; tp /= np.linalg.norm(tp)
    e = overlap * t + np.sqrt(1 - overlap ** 2) * tp
    idents = ["neutral", "male", "female"] + [f"{g}+{a}" for a in ("old", "rich") for g in ("female", "male")]
    eff = {s: (0.0 if s == "neutral" else rng.normal(1.5, 0.7)) for s in idents}
    items, X = [], []
    for i in range(n_items):
        base = 3.0 * rng.standard_normal() * t + 0.3 * rng.standard_normal(d)
        for s in idents:
            items.append(dict(context=ctx, item_id=i, identity="+".join(sorted(s.split("+")))))
            X.append(base + eff[s] * e + 0.05 * rng.standard_normal(d))
    return items, np.array(X), t


def task_var_removed(edit, X, items):
    neu = np.array([x for x, it in zip(X, items) if it["identity"] == "neutral"])
    d = edit.delta(torch.as_tensor(neu, dtype=torch.float32)).numpy()
    dc = d - d.mean(0)
    nc = neu - neu.mean(0)
    return float((dc ** 2).sum() / (nc ** 2).sum())


def effect_left(edit, X, items):
    Xe = edit.apply(torch.as_tensor(X, dtype=torch.float32), 1.0).numpy()
    D0, D1 = P.paired_effects(X, items), P.paired_effects(Xe, items)
    return float((D1 ** 2).sum() / (D0 ** 2).sum())


def test_proposition():
    items, X, _ = toy()
    e = P.fit_pace(X, items, k=2)
    U = e.L @ (e.R @ e.L) if False else None
    # 같은 U로 M = I (직교 사영) 편집
    e_eye = P.fit_pace(X, items, k=2, metric="eye")
    # 무작위 유효 읽기 B: BᵀU = I 를 만족하는 다른 B (같은 범위 span(U_task))
    L = e.L.astype(np.float64); Rm = e.R.astype(np.float64)       # L = U(UᵀMU)⁻¹, R = UᵀM → 범위 span(L)
    Ub = np.linalg.qr(L)[0]
    Bs = []
    for _ in range(20):
        G = rng.standard_normal((X.shape[1], Ub.shape[1]))
        B = G @ np.linalg.inv(Ub.T @ G)                            # BᵀUb = I
        Bs.append(P.PaceEdit(Ub, B.T, e.mu))
    tv = task_var_removed(e, X, items)
    tv_rand = min(task_var_removed(b, X, items) for b in Bs)
    tv_orth = task_var_removed(P.PaceEdit(Ub, Ub.T, e.mu), X, items)
    print(f"명제: 지우는 과제 분산 PACE {tv:.4f} ≤ 같은 범위 직교 사영 {tv_orth:.4f}, 무작위 읽기 최소 {tv_rand:.4f}")
    assert tv <= tv_orth + 1e-6 and tv <= tv_rand + 1e-6


def test_anchor_and_effect():
    items, X, _ = toy()
    e = P.fit_pace(X, items, k=2)
    neu = np.array([x for x, it in zip(X, items) if it["identity"] == "neutral"])
    h = torch.as_tensor(neu.mean(0)[None], dtype=torch.float32)
    moved = (e.apply(h, 1.0) - h).abs().max().item()
    Xe = e.apply(torch.as_tensor(X, dtype=torch.float32), 1.0).numpy()
    coord = np.abs(P.paired_effects(Xe, items) @ e.R.T).max()
    print(f"앵커: 중립 평균 이동 {moved:.1e} / 효과 차단: 편집 후 짝 효과의 U 좌표 최대 {coord:.1e}")
    assert moved < 1e-3 and coord < 1e-3


def test_toy_tradeoff():
    global _t
    items, X, _t = toy(overlap=0.6)
    rows = {}
    for name, ed in {"pace": P.fit_pace(X, items, k=1),
                     "pace_eye": P.fit_pace(X, items, k=1, metric="eye"),
                     "pace_group": P.fit_pace(X, items, k=1, delta="group")}.items():
        rows[name] = (effect_left(ed, X, items), task_var_removed(ed, X, items))
    dm = R.fit_diffmean(X.astype(np.float32), items)
    dm_e = P.PaceEdit(dm.U, dm.U.T, np.zeros(X.shape[1]))
    rows["diffmean"] = (effect_left(dm_e, X, items), task_var_removed(dm_e, X, items))
    for k, (ef, tv) in rows.items():
        print(f"  {k:10s} 남은 정체성 효과 {ef:.3f}  지운 과제 분산 {tv:.3f}")
    D = P.paired_effects(X, items)
    t_share = float(((D @ _t) ** 2).sum() / (D ** 2).sum())
    print(f"  (과제 방향과 겹쳐 지울 수 없는 몫 = 이론 하한 {t_share:.3f})")
    for eps in (0.1, 1.0, 5.0):
        ed = P.fit_pace(X, items, k=1, eps=eps)
        print(f"  ε={eps:<4} PACE 남은 효과 {effect_left(ed, X, items):.3f}  지운 과제 분산 {task_var_removed(ed, X, items):.4f}")
    assert rows["pace"][1] < 0.1 * rows["diffmean"][1] and rows["pace"][0] < 0.6


def test_routing():
    a1, X1, _ = toy(ctx="jobfair"); a2, X2, _ = toy(ctx="lbox")
    X2 = X2 + 20 * rng.standard_normal(X2.shape[1])               # 맥락 중심을 떨어뜨림
    r = P.fit_pace_routed({"jobfair": X1, "lbox": X2}, {"jobfair": a1, "lbox": a2}, k=1)
    h = torch.as_tensor(np.stack([X1[:3], X2[:3]]), dtype=torch.float32)   # 시퀀스 2개 × 위치 3
    pick = r.route(h).tolist()
    out = r.apply(h, 1.0)
    want = torch.stack([h[0] - r.edits[0].delta(h[0]), h[1] - r.edits[1].delta(h[1])])
    print(f"라우팅: 고른 맥락 {[r.names[j] for j in pick]}, 직접 적용과 차이 {(out - want).abs().max().item():.1e}")
    assert pick == [0, 1] and (out - want).abs().max().item() < 1e-4


if __name__ == "__main__":
    test_proposition()
    test_anchor_and_effect()
    print("장난감 비교 (정체성 효과가 과제 방향과 60% 겹침):")
    test_toy_tradeoff()
    test_routing()
    print("OK")
