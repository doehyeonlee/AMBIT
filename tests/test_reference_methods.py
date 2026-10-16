"""
원 논문 방식 비교군(scripts/reference_methods.py)이 공개 코드와 같은 값을 내는지 점검 (CPU, 몇 초)
  python -m tests.test_reference_methods

1) leace_ref : α = 1에서 concept_erasure.LeaceEraser 출력과 같고, 지운 뒤 정체성과의 교차 공분산이 0
2) km_ace    : llm_bias의 get_mean_diff_activations + get_projection_ablation_hook과 같은 값
3) arditi    : refusal_direction의 hook을 같은 위치에 건 모델과 logits가 같고, 모든 층 잔차에서 방향 성분이 0
4) caa       : 마지막 위치만 바뀜, 벡터 = 문항별 (정체성 − 중립) 평균
5) splice    : 정체성 교차 공분산 0, 과제 y와의 공분산 보존, 표본 < 차원이면 데이터 밖 방향은 그대로,
               (원 코드가 있으면) proj.opt_sep_proj와 같은 값

원 저장소와 비교하는 2)·3)은 저장소를 내려받아 경로를 주면 실행되고, 없으면 건너뛴다:
  git clone https://github.com/adamkarvonen/llm_bias      &&  export LLM_BIAS_PATH=$PWD/llm_bias
  git clone https://github.com/andyrdt/refusal_direction  &&  export REFUSAL_DIRECTION_PATH=$PWD/refusal_direction
  SPLICE 원 코드(opt-separation) 폴더                     →  export SPLICE_PATH=<폴더>
"""
import os, sys
import numpy as np, torch
from scripts import reference_methods as F, removal_methods as R

torch.manual_seed(0)
rng = np.random.default_rng(0)


def toy_items(n_items=6, d=32):
    """AMBIT 문항 구조를 흉내 낸 작은 데이터: 문항마다 중립 1 + 성별 2 + 성별×속성(3개) 6."""
    idents = ["neutral", "male", "female"] + [f"{g}+{a}" for a in ("old", "rich", "gay") for g in ("male", "female")]
    items, X = [], []
    base_dir = {a: rng.standard_normal(d) for a in ("female", "old", "rich", "gay")}
    for i in range(n_items):
        mu = rng.standard_normal(d)
        for s in idents:
            x = mu + 0.3 * rng.standard_normal(d)
            for p in s.split("+"):
                if p in base_dir:
                    x = x + base_dir[p]
            items.append(dict(context="jobfair", item_id=i, identity="+".join(sorted(s.split("+")))))
            X.append(x)
    return items, np.array(X, dtype=np.float32)


def test_leace_ref():
    from concept_erasure import LeaceEraser
    items, X = toy_items(n_items=20)
    e = F.fit_leace_ref(X, items)
    ids = sorted({it["identity"] for it in items} - {"neutral"})
    Z = torch.tensor([[it["identity"] == c for c in ids] for it in items], dtype=torch.float64)
    ref = LeaceEraser.fit(torch.as_tensor(X, dtype=torch.float64), Z)
    h = torch.as_tensor(X)
    ours = e.apply(h, 1.0).double()
    theirs = ref(h.double())
    cov = ((ours - ours.mean(0)).T @ (Z - Z.mean(0)) / len(Z)).abs().max().item()
    diff = (ours - theirs).abs().max().item()
    print(f"leace_ref: 공식 eraser와 최대 차이 {diff:.1e}, 지운 뒤 정체성 교차 공분산 최대 {cov:.1e}")
    assert diff < 1e-3 and cov < 1e-3


def test_km_ace():
    items, X = toy_items()
    e = F.fit_km_ace(X, items)
    types = F.km_bias_types(items)
    print(f"km_ace: 방향 {len(types)}개 ({', '.join(types)})")
    path = os.environ.get("LLM_BIAS_PATH")
    if not path:
        print("  (LLM_BIAS_PATH 없음: 원 코드 비교 건너뜀)"); return
    sys.path.insert(0, path)
    import mypkg.pipeline.infra.model_inference as mi
    import mypkg.whitebox_infra.intervention_hooks as ih
    pos = {0: {k: torch.as_tensor(X[p]) for k, (p, n) in types.items()}}
    neg = {0: {k: torch.as_tensor(X[n]) for k, (p, n) in types.items()}}
    mi.get_pos_neg_activations = lambda model, dl, layers: (pos, neg)    # 활성값 수집만 대체, 방향 계산은 원 코드
    dirs = mi.get_mean_diff_activations(None, None, [0])[0]
    U = torch.stack([dirs[k]["diff_acts_D"] for k in types])
    b = torch.stack([dirs[k]["mu"] for k in types])
    hook = ih.get_projection_ablation_hook(U, b)
    h = torch.as_tensor(X)[None]
    theirs = hook(None, None, (h.clone(),))[0][0]
    ours = e.apply(h, 1.0)[0]
    du = (torch.as_tensor(e.U) - U).abs().max().item()
    diff = (ours - theirs).abs().max().item()
    print(f"  llm_bias 원 코드와 방향 차이 {du:.1e}, 편집 결과 차이 {diff:.1e}")
    assert du < 1e-5 and diff < 1e-3


def tiny(kind):
    from transformers import LlamaConfig, LlamaForCausalLM, Gemma2Config, Gemma2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
    c = dict(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
             num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
    return {"llama": lambda: LlamaForCausalLM(LlamaConfig(**c)),
            "qwen3": lambda: Qwen3ForCausalLM(Qwen3Config(**c)),
            "gemma2": lambda: Gemma2ForCausalLM(Gemma2Config(**c, sliding_window=16))}[kind]().eval()


def test_arditi():
    from scripts.removal_models import get_layers
    ids = torch.randint(0, 100, (2, 7))
    r = rng.standard_normal(32).astype(np.float32)
    for kind in ("llama", "qwen3", "gemma2"):
        m = tiny(kind)
        abl = F.ArditiAblation(r, 0)
        resid = []
        hs = abl.register(m, 1.0)
        hs += [b.register_forward_hook(lambda mod, i, o: resid.append(o[0] if isinstance(o, tuple) else o))
               for b in get_layers(m)]
        with torch.no_grad():
            ours = m(ids).logits
        for h in hs:
            h.remove()
        rn = torch.as_tensor(abl.r)
        comp = max((x @ rn).abs().max().item() for x in resid)
        print(f"arditi {kind:7s}: 모든 층 출력의 방향 성분 최대 {comp:.1e}")
        assert comp < 1e-4
        path = os.environ.get("REFUSAL_DIRECTION_PATH")
        if path and kind in ("llama", "qwen3"):     # 원 코드가 지원하는 구조 (attn/mlp 출력이 그대로 잔차에 더해짐)
            sys.path.insert(0, path)
            from pipeline.utils.hook_utils import get_direction_ablation_input_pre_hook, get_direction_ablation_output_hook
            d = torch.as_tensor(r)
            hs = []
            for b in get_layers(m):
                hs.append(b.register_forward_pre_hook(get_direction_ablation_input_pre_hook(direction=d)))
                hs.append(b.self_attn.register_forward_hook(get_direction_ablation_output_hook(direction=d)))
                hs.append(b.mlp.register_forward_hook(get_direction_ablation_output_hook(direction=d)))
            with torch.no_grad():
                theirs = m(ids).logits
            for h in hs:
                h.remove()
            diff = (ours - theirs).abs().max().item()
            print(f"  refusal_direction 원 hook과 logits 차이 {diff:.1e}")
            assert diff < 1e-4
    if not os.environ.get("REFUSAL_DIRECTION_PATH"):
        print("  (REFUSAL_DIRECTION_PATH 없음: 원 코드 비교 건너뜀)")


def test_caa():
    items, X = toy_items()
    c = F.caa_candidates({3: X}, items)[3]
    base = {it["item_id"]: x for x, it in zip(X, items) if it["identity"] == "neutral"}
    want = np.mean([x - base[it["item_id"]] for x, it in zip(X, items) if it["identity"] != "neutral"], 0)
    h = torch.randn(2, 5, X.shape[1])
    out = c.apply(h, 1.0)
    unchanged = (out[:, :-1] - h[:, :-1]).abs().max().item()
    last = (h[:, -1] - out[:, -1] - torch.as_tensor(want)).abs().max().item()
    print(f"caa: 앞 위치 변화 {unchanged:.1e}, 마지막 위치 = h − v 오차 {last:.1e}")
    assert unchanged == 0 and last < 1e-5 and np.abs(c.v - want).max() < 1e-5


def test_splice():
    items, X = toy_items(n_items=40, d=24)                     # 360행 > 24차원 (공분산 정칙)
    item_q = {i: rng.standard_normal() for i in range(40)}
    y = np.array([item_q[it["item_id"]] for it in items])
    X = X + np.outer(y, rng.standard_normal(24))              # 과제 신호를 활성값에 심는다
    e = F.fit_splice(X, items, y)
    ids = sorted({it["identity"] for it in items} - {"neutral"})
    Z = np.array([[it["identity"] == c for c in ids] for it in items], float)
    Xp = e.apply(torch.as_tensor(X), 1.0).numpy().astype(np.float64)
    cz = np.abs(F._cov(Xp, Z)).max()
    cy = np.abs(F._cov(Xp, y) - F._cov(X.astype(np.float64), y)).max()
    print(f"splice: 지운 뒤 정체성 교차 공분산 최대 {cz:.1e}, 과제 공분산 변화 최대 {cy:.1e}")
    assert cz < 1e-3 and cy < 1e-3
    items2, X2 = toy_items(n_items=3, d=64)                    # 27행 < 64차원
    e2 = F.fit_splice(X2, items2, rng.standard_normal(len(items2)))
    _, _, Vt = np.linalg.svd(X2 - X2.mean(0))
    out = Vt[-1].astype(np.float32)                            # 학습 데이터가 덮지 않는 방향
    h = torch.as_tensor(X2.mean(0) + 5 * out)[None]
    moved = (e2.apply(h, 1.0)[0] - h[0]).abs().max().item()
    print(f"  표본 < 차원: 데이터 밖 방향 변화 {moved:.1e} (원 코드는 이 성분을 0으로 만든다)")
    assert moved < 1e-3
    path = os.environ.get("SPLICE_PATH")
    if not path:
        print("  (SPLICE_PATH 없음: 원 코드 비교 건너뜀)"); return
    sys.path.insert(0, path)
    from proj import proj
    pr = proj()
    pr.fit(X.astype(np.float64), Z, y.reshape(-1, 1), method="opt-sep-proj")
    theirs = pr.apply_projection(X.astype(np.float64))
    diff = np.abs(theirs - Xp).max()
    print(f"  원 코드 proj.opt_sep_proj와 최대 차이 {diff:.1e}")
    assert diff < 1e-3


if __name__ == "__main__":
    test_leace_ref()
    test_km_ace()
    test_arditi()
    test_caa()
    test_splice()
    print("OK")
