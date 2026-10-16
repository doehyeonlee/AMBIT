"""저장된 removal JSON의 점수로 판정 v2를 다시 계산한다 (모델을 다시 돌리지 않음).
  python -m scripts.regate outputs/removal outputs/removal/regate_v2.json
v1으로 돌린 결과 파일도 각 칸의 점수(scores)와 대조군 점수가 저장돼 있어 그대로 재판정할 수 있다.
프로브는 결과 파일에 있는 것을 그대로 쓴다(v1 실행은 가장 센 α에서만 프로브)."""
import json, sys, glob, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from scripts import removal_methods as R
from scripts.run_removal import load_context

def test_items(meta):
    items = [it for c in meta["contexts"] for it in load_context(c, meta["n_train"], meta["n_test"], None, meta["seed"])]
    return [it for it in items if it["split"] == "test"]

def regate(path, n_boot=1000):
    r = json.load(open(path)); meta = r["meta"]; T = test_items(meta)
    base = np.array(r["base"]); assert len(base) == len(T), (len(base), len(T))
    idx = {c: [i for i, it in enumerate(T) if it["context"] == c] for c in meta["contexts"]}
    out = {"model": meta["model"], "base": r["disparity_base"], "cells": {}}
    for c, ii in idx.items():
        items = [T[i] for i in ii]; b = base[ii]
        d0 = R.deviations(b, items)
        ctrl = [(R.compression(b, np.array(s)[ii]), R.deviations(np.array(s)[ii], items)) for s in r["controls"].values()]
        for key, cell in r["runs"].items():
            s = np.array(cell["scores"])[ii]
            dv = R.deviations(s, items); C = R.compression(b, s)
            g = R.gate_v2(d0, dv, C, ctrl, n_boot=n_boot)
            pr = cell["eval"][c].get("probe")
            g["probe"] = pr
            g["verdict"] = R.verdict_v2(g, pr)
            out["cells"].setdefault(key, {})[c] = g
    return out

if __name__ == "__main__":
    res = [regate(p) for p in sorted(glob.glob(sys.argv[1] + "/*.json"))]
    json.dump(res, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
