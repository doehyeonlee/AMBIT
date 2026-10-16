"""
여러 모델을 GPU마다 하나씩 동시에 돌리는 실행기

작업(모델 × 데이터셋)을 큐에 넣고, 빈 GPU가 생길 때마다 CUDA_VISIBLE_DEVICES로 GPU를 지정해 실행한다.
27B 모델처럼 한 장(40GB)에 안 들어가는 모델은 GPU 2장을 묶어 주고, 4B처럼 작은 모델은 한 장에 2개씩 올린다.
결과 파일이 이미 있으면 건너뛰므로, 중단돼도 같은 명령을 다시 실행하면 남은 작업만 돈다.

모델은 키 또는 세트 이름으로 지정한다 (scripts/removal_models.py의 MODEL_SETS):
  core     = gemma-2-9b, gemma-3-4b, gemma-3-12b, qwen-3-8b   ← 세 실험 공통 기본 세트
  extended = core + llama-3.1-8b, mistral-7b, gemma-3-27b, qwen-3.5-9b, qwen-3.5-27b

Usage:
  # 네 단계를 같은 모델 세트로 순서대로: SAE 점검 → 행동 측정 → SAE 분석 → 제거
  python -m scripts.run_parallel all --models core

  # 단계별
  python -m scripts.run_parallel check    --models core                 # SAE ↔ 모델 연계성
  python -m scripts.run_parallel behavior --models core --datasets jobfair,lbox,mind
  python -m scripts.run_parallel sae      --models core --datasets jobfair,lbox,mind
  python -m scripts.run_parallel removal  --models core --extra "--methods pca,leace,diffmean,affine,sae,leace_ref,km_ace,arditi,caa --alphas 0.5,1.0,2.0"

  # 계획만 보기
  python -m scripts.run_parallel all --models core --dry-run

로그: logs/<task>_<model>[_<dataset>].log    요약: logs/parallel_<task>.json
"""
from __future__ import annotations
import argparse, json, os, shlex, subprocess, sys, time
from pathlib import Path

from scripts.removal_models import REMOVAL_MODELS, expand_models

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
STAGES = ["check", "behavior", "sae", "removal"]
DEFAULT_REMOVAL_EXTRA = "--methods pca,leace,diffmean,affine,sae,leace_ref,km_ace,arditi,caa --alphas 0.5,1.0,2.0"

# 한 장(40GB A100 기준)에 안 들어가는 모델 → 필요한 GPU 수
GPUS_NEEDED = {"gemma-3-27b": 2, "qwen-3.5-27b": 2}


# 작은 모델 → GPU 한 장에 동시에 올릴 작업 수 (gemma-3-4b ≈ 9GB)
GPU_SHARE = {"gemma-3-4b": 2}


def gpus_needed(model: str, gpu_mem_gb: int) -> int:
    return 1 if gpu_mem_gb >= 80 else GPUS_NEEDED.get(model, 1)


def gpu_share(model: str, gpu_mem_gb: int) -> int:
    """GPU 한 장을 나눠 쓸 수 있는 작업 수 (1 = 혼자 씀)."""
    return GPU_SHARE.get(model, 1) if gpus_needed(model, gpu_mem_gb) == 1 else 1


def has_sae(model: str) -> bool:
    return bool((REMOVAL_MODELS.get(model) or {}).get("sae"))


def build_jobs(task: str, models: list[str], datasets: list[str], extra: str) -> list[dict]:
    jobs, py = [], [sys.executable, "-m"]
    ex = shlex.split(extra) if extra else []
    for m in models:
        if task in ("behavior", "sae") and m.endswith("-ptsae"):
            print(f"  ({task}) {m}: SAE 비교용 별칭이라 제외 (check·removal에서만 사용)")
            continue
        if task in ("check", "sae") and not has_sae(m):
            print(f"  ({task}) {m}: 공개 SAE가 없어 제외")
            continue
        if task == "check":
            jobs.append(dict(name=f"check_{m}", model=m, cmd=py + ["scripts.check_sae", "--model", m] + ex,
                             done=PROJECT_ROOT / "outputs" / "sae_check" / f"{m}.json"))
        elif task == "removal":
            jobs.append(dict(name=f"removal_{m}", model=m, cmd=py + ["scripts.run_removal", "--model", m] + ex,
                             done=None))   # run_removal이 끝난 조건을 스스로 건너뛴다
        else:
            for ds in datasets:
                data = f"data/{ds}.csv"
                if task == "behavior":
                    done = PROJECT_ROOT / "outputs" / "local" / ds / f"raw_{ds}_{m}.json"
                    cmd = py + ["scripts.run_one_experiment", "--data-file", data, "--model", m] + ex
                else:   # sae: 층별 결과 폴더가 여러 개라 완료 판정은 생략
                    done = None
                    cmd = py + ["scripts.run_sae_analysis", "--data-file", data, "--model", m, "--multi-layer"] + ex
                jobs.append(dict(name=f"{task}_{m}_{ds}", model=m, cmd=cmd, done=done))
    return jobs


def detect_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.strip().splitlines()
        rows = [tuple(map(int, l.split(","))) for l in out]
        return [str(i) for i, _ in rows], min(m for _, m in rows) // 1024
    except Exception:
        return [], 0


def run_stage(task, models, datasets, extra, gpus, mem, force=False, dry_run=False) -> bool:
    jobs = build_jobs(task, models, datasets, extra)
    todo = [j for j in jobs if force or not (j["done"] and Path(j["done"]).exists())]
    skipped = len(jobs) - len(todo)
    todo.sort(key=lambda j: -gpus_needed(j["model"], mem))   # 큰 모델을 먼저 배치
    print(f"\n=== {task}: 작업 {len(todo)}개" + (f", 완료돼서 건너뜀 {skipped}개" if skipped else "") + " ===")
    for j in todo:
        share = gpu_share(j["model"], mem)
        print(f"  [{gpus_needed(j['model'], mem)} GPU{f' 1/{share}' if share > 1 else ''}] {j['name']}: {' '.join(j['cmd'][2:])}")
    if dry_run or not todo:
        return True

    LOG_DIR.mkdir(exist_ok=True)
    # GPU별 점유율 (1.0 = 꽉 참). 작은 모델은 1/share씩 차지해 한 장에 여러 개가 같이 돈다
    load, running, summary, t0 = {g: 0.0 for g in gpus}, [], {}, time.time()
    while todo or running:
        for r in running[:]:
            rc = r["proc"].poll()
            if rc is None:
                continue
            running.remove(r)
            for g in r["gpus"]:
                load[g] -= r["weight"]
            dt = (time.time() - r["start"]) / 60
            summary[r["job"]["name"]] = dict(returncode=rc, minutes=round(dt, 1), gpus=r["gpus"], log=str(r["log"]))
            print(f"{'완료' if rc == 0 else '실패'} {r['job']['name']} (GPU {','.join(r['gpus'])}, {dt:.0f}분)"
                  + ("" if rc == 0 else f" → 로그: {r['log']}"), flush=True)
            json.dump(summary, open(LOG_DIR / f"parallel_{task}.json", "w"), ensure_ascii=False, indent=1)
        for j in todo[:]:
            need = gpus_needed(j["model"], mem)
            if need > len(gpus):
                print(f"건너뜀 {j['name']}: GPU {need}장 필요"); todo.remove(j); continue
            weight = 1.0 / gpu_share(j["model"], mem)
            if need > 1:
                idle = [g for g in gpus if load[g] < 1e-6]
                use = idle[:need] if len(idle) >= need else None
            else:
                # 이미 일부 쓰는 GPU부터 채워서 빈 GPU를 큰 모델용으로 남긴다
                fits = [g for g in gpus if load[g] + weight <= 1 + 1e-6]
                use = [max(fits, key=lambda g: load[g])] if fits else None
            if use:
                for g in use:
                    load[g] += weight
                log = LOG_DIR / f"{j['name']}.log"
                env = os.environ | {"CUDA_VISIBLE_DEVICES": ",".join(use), "PYTHONUNBUFFERED": "1",
                                    "TOKENIZERS_PARALLELISM": "false"}
                proc = subprocess.Popen(j["cmd"], cwd=PROJECT_ROOT, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
                running.append(dict(job=j, proc=proc, gpus=use, weight=weight, log=log, start=time.time()))
                todo.remove(j)
                print(f"시작 {j['name']} → GPU {','.join(use)}  (로그 {log.relative_to(PROJECT_ROOT)})", flush=True)
        time.sleep(5)
    ok = sum(v["returncode"] == 0 for v in summary.values())
    print(f"=== {task}: {len(summary)}개 중 성공 {ok}개, {(time.time() - t0) / 60:.0f}분 ===")
    return ok == len(summary)


def main():
    ap = argparse.ArgumentParser(description="여러 모델 동시 실행 (GPU당 작업 1개)")
    ap.add_argument("task", choices=STAGES + ["all"])
    ap.add_argument("--models", default="core", help="모델 키나 세트 이름(core, extended)을 쉼표로")
    ap.add_argument("--datasets", default="jobfair,lbox,mind", help="behavior·sae용")
    ap.add_argument("--gpus", default=None, help="사용할 GPU 번호 (기본: 전부)")
    ap.add_argument("--gpu-mem", type=int, default=None, help="GPU 한 장 메모리(GB). 기본: nvidia-smi로 감지")
    ap.add_argument("--extra", default="", help="각 작업에 덧붙일 인자 (단일 단계용)")
    ap.add_argument("--removal-extra", default=DEFAULT_REMOVAL_EXTRA, help="all에서 removal 단계 인자")
    ap.add_argument("--keep-going", action="store_true", help="all: 앞 단계 실패가 있어도 다음 단계 진행")
    ap.add_argument("--force", action="store_true", help="결과 파일이 있어도 다시 실행")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    found, mem = detect_gpus()
    gpus = a.gpus.split(",") if a.gpus else found
    mem = a.gpu_mem or mem or 40
    if not gpus and not a.dry_run:
        sys.exit("GPU를 찾지 못했습니다 (--gpus로 지정 가능)")
    models = expand_models(a.models)
    unknown = [m for m in models if m not in REMOVAL_MODELS]
    if unknown:
        sys.exit(f"알 수 없는 모델: {unknown}. 사용 가능: {list(REMOVAL_MODELS)}")
    print(f"모델 {models}, GPU {gpus or '(dry-run)'} ({mem}GB/장)")

    stages = STAGES if a.task == "all" else [a.task]
    for st in stages:
        extra = a.removal_extra if (a.task == "all" and st == "removal") else (a.extra if a.task != "all" else "")
        ok = run_stage(st, models, a.datasets.split(","), extra, gpus, mem, a.force, a.dry_run)
        if not ok and a.task == "all" and not a.keep_going:
            sys.exit(f"{st} 단계에 실패한 작업이 있어 멈춥니다 (logs/ 확인, --keep-going으로 계속 가능)")
        if st == "check" and not a.dry_run:
            _print_check_summary(models)


def _print_check_summary(models):
    print("\nSAE 연계성 요약 (outputs/sae_check/):")
    for m in models:
        p = PROJECT_ROOT / "outputs" / "sae_check" / f"{m}.json"
        if not p.exists():
            continue
        rep = json.load(open(p))
        cells = [f"L{L}:{r['verdict']}({r['raw']['ev']:.2f}/{r['chat']['ev']:.2f})" if "raw" in r else f"L{L}:{r['verdict']}"
                 for L, r in rep["layers"].items()]
        print(f"  {m:14s} " + "  ".join(cells))
    print("  (EV raw/chat; FAIL이 있는 층의 SAE 결과는 쓰지 말 것)")


if __name__ == "__main__":
    main()
