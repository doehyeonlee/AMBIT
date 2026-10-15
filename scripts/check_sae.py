"""
SAE ↔ 모델 연계성 점검 (GPU에서 실행)

모델의 실제 활성값으로 각 SAE가 얼마나 잘 맞는지 확인한다. 2부(ΔnFEP)와 3부(SAE 소거)를 돌리기 전에 실행.

  python -m scripts.check_sae --model gemma-3-12b
  python -m scripts.run_parallel check --models core        # 네 모델 동시에

점검 항목 (층마다)
  - 차원: SAE d_model == 모델 hidden size
  - 재구성 설명 분산(EV) = 1 − ‖x − x̂‖² / ‖x − x̄‖²   (프롬프트 전체 토큰, 첫 토큰·패딩 제외)
  - 마지막 토큰 EV (2부·3부가 실제로 읽는 위치)
  - 평균 L0 (활성 특징 수), 죽은 토큰 비율(활성 특징 0개)
  - 형식 두 가지: raw(2부 run_sae_analysis 방식) / chat(3부 run_removal 방식, 채팅 템플릿)
판정: EV ≥ 0.7 OK, 0.5 ~ 0.7 WARN(해석 주의), < 0.5 FAIL(해당 SAE 결과는 쓰지 말 것)

출력: outputs/sae_check/<model>.json (+ 표 출력)
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch

from scripts import removal_models as M
from scripts.run_one_experiment import SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "sae_check"
torch.set_grad_enabled(False)


def sample_prompts(n_per_ctx=16, seed=0):
    """맥락마다 문항·정체성을 고르게 섞어 n개씩."""
    rng = np.random.default_rng(seed)
    out = []
    for ctx in ("jobfair", "lbox", "mind"):
        rows = list(csv.DictReader(open(PROJECT_ROOT / "data" / f"{ctx}.csv", encoding="utf-8-sig")))
        for i in rng.choice(len(rows), n_per_ctx, replace=False):
            out.append((ctx, rows[int(i)]["Prompt"].strip()))
    return out


def collect(model, tok, texts, layers, batch=4, max_len=4096, add_special=True):
    """층별 (전체 토큰 활성값, 마지막 토큰 활성값). 왼쪽 패딩, 첫 실제 토큰(BOS)과 패딩 제외."""
    blocks = M.get_layers(model)
    store = {l: [] for l in layers}
    last = {l: [] for l in layers}
    mask_holder = {}

    def mk(l):
        def hook(mod, inp, out):
            hs = (out[0] if isinstance(out, tuple) else out).float()
            m = mask_holder["m"]
            store[l].append(hs[m].cpu())
            last[l].append(hs[:, -1].cpu())
        return hook

    tok.padding_side, tok.truncation_side = "left", "left"
    dev = next(model.parameters()).device
    hs = [blocks[l].register_forward_hook(mk(l)) for l in layers]
    try:
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len, add_special_tokens=add_special).to(dev)
            am = enc["attention_mask"].bool()
            first = am.float().argmax(-1)                      # 첫 실제 토큰 위치
            m = am.clone()
            m[torch.arange(len(m)), first] = False             # BOS/첫 토큰 제외 (노름이 매우 큼)
            mask_holder["m"] = m
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], position_ids=pos, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return {l: torch.cat(store[l]) for l in layers}, {l: torch.cat(last[l]) for l in layers}


def score(sae, X, chunk=2048):
    X = X.to(sae.device, sae.dtype)
    res = tot = 0.0
    l0, dead = [], 0
    mu = X.mean(0)
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        f = sae.encode(x)
        xh = sae.decode(f)
        res += float(((x - xh) ** 2).sum())
        tot += float(((x - mu) ** 2).sum())
        nz = (f != 0).sum(-1)
        l0.append(nz.float().cpu())
        dead += int((nz == 0).sum())
    l0 = torch.cat(l0)
    return {"ev": 1 - res / tot, "l0": float(l0.mean()), "dead_frac": dead / len(X), "n_tokens": len(X)}


def verdict(ev):
    return "OK" if ev >= 0.7 else ("WARN" if ev >= 0.5 else "FAIL")


def main():
    ap = argparse.ArgumentParser(description="SAE ↔ 모델 연계성 점검")
    ap.add_argument("--model", required=True, choices=[k for k, v in M.REMOVAL_MODELS.items() if v.get("sae")])
    ap.add_argument("--layers", default=None, help="쉼표 목록 (기본: 2부 층 + 3부 소거 층)")
    ap.add_argument("--n-per-ctx", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()

    tok, model = M.load_model(a.model)
    cfg = model.config
    d_model = (cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg).hidden_size
    layers = sorted({int(x) for x in a.layers.split(",")} if a.layers else
                    set(M.sae_layers(a.model)) | {M.sae_mid_layer(a.model, model)})
    prompts = sample_prompts(a.n_per_ctx)
    raw = [p for _, p in prompts]
    chat = [M.format_prompt(tok, a.model, p, SYSTEM_PROMPT) for _, p in prompts]
    bos = tok.bos_token or ""
    t = time.time()
    acts = {"raw": collect(model, tok, raw, layers, a.batch),
            "chat": collect(model, tok, chat, layers, a.batch, add_special=not (bos and chat[0].startswith(bos)))}
    print(f"[{a.model}] 층 {M.n_layers(model)}개, hidden {d_model}, 점검 층 {layers}, 프롬프트 {len(prompts)}개 ({time.time() - t:.0f}s)")

    report = dict(model=a.model, hf=M.REMOVAL_MODELS[a.model]["hf"], hidden_size=d_model,
                  n_layers=M.n_layers(model), sae=M.REMOVAL_MODELS[a.model]["sae"].get("repo"), layers={})
    dev = str(next(model.parameters()).device)
    for L in layers:
        sae = M.load_sae(a.model, L, device=dev)
        d_sae_in = sae.W_dec.shape[1]
        row = dict(source=sae.meta["source"], sae_layer=sae.meta["layer"], trained_on=sae.meta["trained_on"],
                   d_model_ok=bool(d_sae_in == d_model), d_sae=int(sae.W_dec.shape[0]))
        if sae.meta["layer"] != L:
            row["note"] = f"요청 층 {L}에 SAE가 없어 {sae.meta['layer']}층 사용"
        if row["d_model_ok"]:
            for fmt in ("raw", "chat"):
                allX, lastX = acts[fmt][0][L], acts[fmt][1][L]
                row[fmt] = score(sae, allX) | {"ev_last_token": score(sae, lastX)["ev"]}
            row["verdict"] = verdict(min(row["raw"]["ev"], row["chat"]["ev"]))
        else:
            row["verdict"] = "FAIL"
            row["note"] = f"차원 불일치: SAE {d_sae_in} vs 모델 {d_model}"
        report["layers"][str(L)] = row
        del sae
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    out_dir = Path(a.output_dir) if a.output_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out_dir / f"{a.model}.json", "w"), ensure_ascii=False, indent=1)

    print(f"\n{'층':>4} {'학습':>4} {'EV raw':>7} {'EV chat':>8} {'EV 마지막(raw/chat)':>20} {'L0 raw':>7} {'판정':>5}  SAE")
    for L, r in report["layers"].items():
        if "raw" in r:
            print(f"{L:>4} {r['trained_on']:>4} {r['raw']['ev']:7.3f} {r['chat']['ev']:8.3f} "
                  f"{r['raw']['ev_last_token']:9.3f}/{r['chat']['ev_last_token']:<9.3f} {r['raw']['l0']:7.0f} {r['verdict']:>5}  {r['source']}")
        else:
            print(f"{L:>4} {r['trained_on']:>4} {'-':>7} {'-':>8} {'-':>20} {'-':>7} {r['verdict']:>5}  {r.get('note', '')}")
    print(f"\n저장: {out_dir / (a.model + '.json')}")


if __name__ == "__main__":
    main()
