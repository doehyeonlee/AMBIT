"""
3부(제거): 맥락 간 전이 행렬 + 압축 통제 판정 + 표현 잔존 프로브

한 맥락의 학습 문항으로 제거 방향을 추정하고, 세 맥락(JobFair · LBOX · Mind)의 평가 문항에서
격차 감소가 "같은 압축률의 무관 대조군"보다 큰지(E = ΔD − ΔD_ctrl(C), 부트스트랩 CI)를 판정한다.

Usage:
  # 10분 점검
  python -m scripts.run_removal --model gemma-3-4b --n-train 4 --n-test 4 --methods diffmean --alphas 1.0
  # 본 실행
  python -m scripts.run_removal --model gemma-3-12b --methods pca,leace,diffmean,affine,sae
  # 원 논문 공개 코드 방식 비교군 (scripts/reference_methods.py)
  python -m scripts.run_removal --model gemma-3-12b --methods leace_ref,km_ace,arditi,caa
  python -m scripts.run_removal --list-models

입력: data/{jobfair,lbox,mind}.csv (53행 = 문항 하나: baseline 1 + 성별 2 + 성별×속성 50)
출력: outputs/removal/<model>.json  — 조건 하나가 끝날 때마다 저장, 다시 실행하면 끝난 조건은 건너뜀
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np

from scripts import removal_models as M
from scripts import removal_methods as R
from scripts import reference_methods as F
from scripts.run_one_experiment import SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "removal"
BLOCK = 53
FITTERS = {"pca": R.fit_pca, "leace": R.fit_leace, "diffmean": R.fit_diffmean,
           "affine": R.fit_affine, "steer": R.fit_steer, "leace_ref": F.fit_leace_ref}
REFERENCE = {"leace_ref", "km_ace", "arditi", "caa"}   # 원 논문 공개 코드 방식 (reference_methods.py)


def load_context(ctx, n_train, n_test, identities, seed):
    """CSV → 문항 dict 리스트. 53행 블록이 한 문항(item_id)이고, 문항 단위로 train/test를 나눈다."""
    rows = list(csv.DictReader(open(PROJECT_ROOT / "data" / f"{ctx}.csv", encoding="utf-8-sig")))
    assert len(rows) % BLOCK == 0 and all(rows[i]["demographic_identifier"] == "[]" for i in range(0, len(rows), BLOCK)), \
        f"{ctx}.csv가 53행 블록 구조가 아닙니다"
    n_items = len(rows) // BLOCK
    perm = np.random.default_rng(seed).permutation(n_items)
    split = {int(i): "train" for i in perm[:n_train]} | {int(i): "test" for i in perm[n_train:n_train + n_test]}
    items = []
    for i, r in enumerate(rows):
        iid = i // BLOCK
        if iid not in split:
            continue
        demo = json.loads(r["demographic_identifier"])
        ident = "+".join(sorted(demo)) if demo else "neutral"
        if identities and ident != "neutral" and ident not in identities:
            continue
        items.append(dict(context=ctx, item_id=iid, identity=ident, split=split[iid], prompt=r["Prompt"].strip()))
    return items


def main():
    ap = argparse.ArgumentParser(description="3부: 제거 기법 × 맥락 간 전이 × 압축 통제 판정")
    ap.add_argument("--model", choices=list(M.REMOVAL_MODELS))
    ap.add_argument("--contexts", default="jobfair,lbox,mind")
    ap.add_argument("--n-train", type=int, default=20, help="맥락당 방향 추정용 문항 수 (문항당 53행)")
    ap.add_argument("--n-test", type=int, default=30, help="맥락당 평가 문항 수")
    ap.add_argument("--identities", default="all", help="'all' 또는 쉼표 목록 (예: male,female,black+female)")
    ap.add_argument("--methods", default="pca,leace,diffmean",
                    help="pca,leace,diffmean,affine,steer,sae / 원 논문 방식: leace_ref,km_ace,arditi,caa")
    ap.add_argument("--alphas", default="0.5,1.0")
    ap.add_argument("--control-alphas", default="0.25,0.5,1.0,2.0")
    ap.add_argument("--band", default="0.43,0.62", help="개입 깊이 비율 (Gemma-2-9B 18~26층 ≈ 0.43~0.62)")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--sae-layer", type=int, default=None,
                    help="SAE 소거 층 (기본: 레지스트리의 mid = 2부 가운데 층. gemma-2-9b 20, gemma-3-4b 17, gemma-3-12b 24, qwen-3-8b 18)")
    ap.add_argument("--sae-feats", type=int, default=32)
    ap.add_argument("--sae-width", default=None, help="Gemma Scope 폭 덮어쓰기 (예: gemma-2-9b 131k)")
    ap.add_argument("--km-types", default="all", choices=["all", "gender"],
                    help="km_ace 방향: all = 성별 + 속성마다 하나씩 (원 코드처럼 이진 속성별), gender = 성별 하나")
    ap.add_argument("--km-layers", default="all", choices=["all", "band"], help="km_ace 개입 층 (원 코드 기본: 전 층)")
    ap.add_argument("--select", default="search", choices=["search", "mid"],
                    help="arditi·caa 방향을 뽑을 층: search = 원 논문처럼 후보 층마다 학습 문항으로 평가해 고름, mid = 개입 구간 가운데")
    ap.add_argument("--select-max-c", type=float, default=0.1,
                    help="search에서 학습 문항 압축률 C가 이 값을 넘는 층은 제외 (Arditi의 KL < 0.1 필터 역할)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--list-models", action="store_true")
    a = ap.parse_args()

    if a.list_models:
        for k, v in M.REMOVAL_MODELS.items():
            sae = v["sae"]
            print(f"  {k:20s} {v['hf']:40s} SAE={'—' if not sae else sae['backend'] + ' (' + sae['trained_on'] + ')'}")
        return
    if not a.model:
        ap.error("--model 이 필요합니다")

    out_dir = Path(a.output_dir) if a.output_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{a.model}.json"
    res = json.load(open(path)) if path.exists() else {}
    save = lambda: json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)

    tok, model = M.load_model(a.model)
    # 'male+black'처럼 순서를 바꿔 써도 되도록 정렬해서 데이터 키와 맞춘다
    idents = None if a.identities == "all" else {"+".join(sorted(x.strip().split("+"))) for x in a.identities.split(",")}
    ctxs = a.contexts.split(",")
    items = [it for c in ctxs for it in load_context(c, a.n_train, a.n_test, idents, a.seed)]
    for it in items:
        it["text"] = M.format_prompt(tok, a.model, it["prompt"], SYSTEM_PROMPT)
    train = {c: [it for it in items if it["context"] == c and it["split"] == "train"] for c in ctxs}
    test = [it for it in items if it["split"] == "test"]
    test_c = {c: [i for i, it in enumerate(test) if it["context"] == c] for c in ctxs}
    lo, hi = map(float, a.band.split(","))
    band = M.depth_band(model, lo, hi, a.step)
    cfg = dict(model=a.model, hf=M.REMOVAL_MODELS[a.model]["hf"], n_layers=M.n_layers(model), band=band,
               contexts=ctxs, n_train=a.n_train, n_test=a.n_test, identities=a.identities, seed=a.seed)
    if res.get("meta", cfg) != cfg:
        sys.exit(f"{path}의 설정이 지금과 다릅니다. --output-dir를 바꾸거나 파일을 옮기세요.\n{res['meta']}")
    res["meta"] = cfg
    print(f"[{a.model}] 층 {cfg['n_layers']}개, 개입 층 {band}, 평가 {len(test)}행, 학습 {sum(map(len, train.values()))}행")

    def score(edits=None, alpha=0.0):
        t = time.time()
        if edits:
            with R.Intervention(model, edits, alpha):
                s = R.score_items(model, tok, test, a.batch)
        else:
            s = R.score_items(model, tok, test, a.batch)
        print(f"   채점 {len(test)}행 {time.time() - t:.0f}s")
        return s

    split = lambda s: {c: (s[idx], [test[i] for i in idx]) for c, idx in test_c.items()}

    # ---- 기준
    base = np.array(res["base"]) if "base" in res else score()
    res["base"] = base.tolist(); save()
    B = split(base)
    dev0 = {c: R.deviations(*B[c]) for c in ctxs}
    res["disparity_base"] = {c: float(R.disparity(dev0[c])) for c in ctxs}
    print("기준 격차:", {c: round(v, 3) for c, v in res["disparity_base"].items()})

    # ---- 대조군: 무작위 방향 / 전체 활성 PCA (정체성과 무관하게 분산만 줄임)
    acts_all = R.collect_acts(model, tok, [it for c in ctxs for it in train[c]], band, a.batch)
    controls = {"random": R.random_controls(acts_all, seed=a.seed), "global_pca": R.global_pca_controls(acts_all)}
    res.setdefault("controls", {})
    for cname, edits in controls.items():
        for al in map(float, a.control_alphas.split(",")):
            key = f"{cname}@{al}"
            if key not in res["controls"]:
                print(f"대조군 {key}")
                res["controls"][key] = score(edits, al).tolist(); save()
    ctrl_runs = {c: [] for c in ctxs}
    for s in res["controls"].values():
        S = split(np.array(s))
        for c in ctxs:
            ctrl_runs[c].append((R.compression(B[c][0], S[c][0]), R.deviations(*S[c])))

    def select_layer(meth, fc, cands):
        """Arditi(방향 선택) · CAA(층 스윕)처럼 후보 층마다 학습 문항에서 α=1로 개입해 격차가 가장 작은 층을 고른다.
        평가 문항은 쓰지 않는다. 압축률이 select-max-c를 넘는 후보는 제외."""
        key, mid = f"{meth}|{fc}", band[len(band) // 2]
        if key in res.get("selected_layer", {}):
            return res["selected_layer"][key]
        if a.select == "mid":
            return mid
        tr = train[fc]
        res.setdefault("train_base", {})
        if fc not in res["train_base"]:
            res["train_base"][fc] = R.score_items(model, tok, tr, a.batch).tolist(); save()
        tb = np.array(res["train_base"][fc])
        log, best = {}, None
        for l, e in cands.items():
            with R.Intervention(model, {l: e}, 1.0):
                s = R.score_items(model, tok, tr, a.batch)
            C, D = R.compression(tb, s), float(R.disparity(R.deviations(s, tr)))
            log[str(l)] = {"C": C, "disparity": D}
            print(f"   {key} 후보 층 {l}: 학습 격차 {D:.3f}, C {C:+.3f}")
            if C <= a.select_max_c and (best is None or D < best[1]):
                best = (l, D)
        res.setdefault("layer_search", {})[key] = log
        return best[0] if best else mid

    # ---- 기법 × 학습 맥락 × 강도
    res.setdefault("runs", {})
    alphas = list(map(float, a.alphas.split(",")))
    sae = None
    for meth in a.methods.split(","):
        for fc in ctxs:
            todo = [al for al in alphas if f"{meth}|{fc}|{al}" not in res["runs"]]
            if not todo:
                continue
            tr = train[fc]
            if meth == "sae":
                if sae is None:
                    L0 = a.sae_layer if a.sae_layer is not None else M.sae_mid_layer(a.model, model)
                    sae = M.load_sae(a.model, L0, device=str(R._device(model)), width=a.sae_width)
                    if sae is None:
                        print(f"{a.model}는 공개 SAE가 없어 sae를 건너뜁니다."); break
                    print(f"SAE: {sae.meta['source']}")
                L = sae.meta["layer"]
                acts = R.collect_acts(model, tok, tr, [L], a.batch)
                fid = M.sae_fidelity(sae, acts[L])
                res.setdefault("sae_fidelity", {})[fc] = fid | {"source": sae.meta["source"]}
                print(f"SAE 층 {L} 재구성 EV={fid['explained_variance']:.2f}, L0={fid['l0']:.0f}"
                      + ("  ← 0.7 미만: SAE 결과 해석 보류" if fid["explained_variance"] < 0.7 else ""))
                edits = {L: R.SAEAblation(sae, R.pick_sae_features(sae, acts[L], tr, a.sae_feats))}
            elif meth == "km_ace":
                km_layers = list(range(cfg["n_layers"])) if a.km_layers == "all" else band
                acts = F.collect_acts_mean(model, tok, tr, km_layers, a.batch)
                edits = {l: F.fit_km_ace(acts[l], tr, a.km_types) for l in km_layers}
                del acts
            elif meth in ("arditi", "caa"):
                cands = (F.arditi_candidates if meth == "arditi" else F.caa_candidates)(
                    R.collect_acts(model, tok, tr, band, a.batch), tr)
                L = select_layer(meth, fc, cands)
                res.setdefault("selected_layer", {})[f"{meth}|{fc}"] = L
                print(f"{meth}|{fc}: 방향 층 {L}" + (" (모든 층에서 소거)" if meth == "arditi" else " (이 층 마지막 위치에 적용)"))
                edits = {max(band): cands[L]} if meth == "arditi" else {L: cands[L]}
            else:
                edits = R.fit_per_layer(FITTERS[meth], R.collect_acts(model, tok, tr, band, a.batch), tr)
            for al in todo:
                key = f"{meth}|{fc}|{al}"
                print(key)
                S = split(score(edits, al))
                cell = {"eval": {}}
                for ec in ctxs:
                    dv = R.deviations(*S[ec])
                    g = R.gate(dev0[ec], dv, R.compression(B[ec][0], S[ec][0]), ctrl_runs[ec])
                    g["disparity_after"] = float(R.disparity(dv))
                    cell["eval"][ec] = g
                cell["scores"] = np.concatenate([S[c][0] for c in ctxs]).tolist()
                if not a.no_probe and al == max(alphas):   # 표현 잔존: 개입을 건 채 마지막 개입 층에서 정체성 디코딩
                    last = min(max(edits), max(band))   # 전 층 개입(km_ace)도 다른 기법과 같은 층에서 잰다
                    with R.Intervention(model, edits, al):
                        for ec in ctxs:
                            sub = [it for it in test if it["context"] == ec and it["identity"] != "neutral"]
                            X = R.collect_acts(model, tok, sub, [last], a.batch)[last]
                            pr = R.probe_acc(X, np.array([it["identity"] for it in sub]))
                            cell["eval"][ec] |= {"probe": pr, "verdict": R.verdict(cell["eval"][ec]["behavior_pass"], pr)}
                res["runs"][key] = cell; save()

    print("\n기법|학습맥락|α  →  평가맥락: E [95% CI] 판정")
    for key, cell in res["runs"].items():
        print(f"{key:26s} " + "  ".join(
            f"{ec}: {g['E_mean']:+.3f} [{g['ci'][0]:+.3f},{g['ci'][1]:+.3f}] {g.get('verdict', '')}"
            for ec, g in cell["eval"].items()))
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
