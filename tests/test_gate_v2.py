"""판정 v2(removal_methods.gate_v2) 점검 — 다섯 가지 status가 의도대로 나오는지 (CPU, 1초)
  python -m tests.test_gate_v2
v1의 결함 재현: 분산을 늘린 대조군(C < 0)이 섞여 있어도, 분산을 늘린 기법은 '분산 증가'로 분리되고
대조군 범위 밖(C > 최대)은 판정하지 않는다."""
import numpy as np
from scripts import removal_methods as R

rng = np.random.default_rng(0)
IDS = ["a", "b", "c"]


def devs(scale):
    return [(g, i, scale * (k - 1) + 0.01 * rng.standard_normal()) for i in range(30) for k, g in enumerate(IDS)]


if __name__ == "__main__":
    d0 = devs(1.0)
    ctrl = [(0.2, devs(0.9)), (0.5, devs(0.6)), (0.8, devs(0.3)), (-3.0, devs(5.0))]   # 마지막: 분산을 늘린 대조군
    cases = {"inflation": (devs(0.5), -0.5), "out_of_range": (devs(0.0), 0.95), "ineffective": (devs(1.2), 0.3),
             "compression": (devs(0.6), 0.5), "pass": (devs(0.2), 0.3)}
    for want, (dv, C) in cases.items():
        g = R.gate_v2(d0, dv, C, ctrl, n_boot=200)
        print(f"{want:13s} → {g['status']:13s} ΔD {g['dD_mean']:+.2f}  E {g['E_mean']:+.2f}  판정 {R.verdict_v2(g, None)}")
        assert g["status"] == want
    print("OK")
