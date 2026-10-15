"""
여러 모델을 GPU마다 하나씩 동시에 돌리는 실행기

작업(모델 × 데이터셋)을 큐에 넣고, 빈 GPU가 생길 때마다 CUDA_VISIBLE_DEVICES로 GPU를 지정해 실행한다.
27B 모델처럼 한 장(40GB)에 안 들어가는 모델은 GPU 2장을 묶어 준다.
결과 파일이 이미 있으면 건너뛰므로, 중단돼도 같은 명령을 다시 실행하면 남은 작업만 돈다.

Usage:
  # 행동 측정: 3개 모델 × 3개 맥락 = 9개 작업을 GPU 4장에 분배
  python -m scripts.run_parallel behavior --models gemma-2-9b,gemma-3-4b,mistral-7b --datasets jobfair,lbox,mind

  # SAE 분석 (층 여러 개)
  python -m scripts.run_parallel sae --models gemma-2-9b,gemma-3-12b,qwen-3-8b --datasets lbox,mind

  # 3부 제거 (모델당 작업 1개, 세 맥락을 한 번에)
  python -m scripts.run_parallel removal --models gemma-3-12b,qwen-3-8b,llama-3.1-8b,gemma-2-9b \
      --extra "--methods pca,leace,diffmean,affine,sae --alphas 0.5,1.0,2.0"

  # 계획만 보기
  python -m scripts.run_parallel behavior --models gemma-2-9b,gemma-3-4b --datasets jobfair --dry-run

로그: logs/<task>_<model>_<dataset>.log    요약: logs/parallel_<task>.json
"""
from __future__ import annotations
import argparse, json, os, shlex, subprocess, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

# 한 장(40GB A100 기준)에 안 들어가는 모델 → 필요한 GPU 수
GPUS_NEEDED = {"gemma-3-27b": 2, "qwen-3.5-27b": 2}


def gpus_needed(model: str, gpu_mem_gb: int) -> int:
    n = GPUS_NEEDED.get(model, 1)
    return 1 if gpu_mem_gb >= 80 else n


def build_jobs(task: str, models: list[str], datasets: list[str], extra: str) -> list[dict]:
    jobs = []
    py = [sys.executable, "-m"]
    ex = shlex.split(extra) if extra else []
    if task == "removal":
        for m in models:
            jobs.append(dict(name=f"removal_{m}", model=m,
                             cmd=py + ["scripts.run_removal", "--model", m] + ex,
                             done=PROJECT_ROOT / "outputs" / "removal" / f"{m}.json", done_key="runs"))
        return jobs
    for m in models:
        for ds in datasets:
            data = f"data/{ds}.csv"
            if task == "behavior":
                done = PROJECT_ROOT / "outputs" / "local" / ds / f"raw_{ds}_{m}.json"
                cmd = py + ["scripts.run_one_experiment", "--data-file", data, "--model", m] + ex
            elif task == "sae":
                done = None   # 층별 결과 폴더가 여러 개라 완료 판정은 run_sae_analysis에 맡김
                cmd = py + ["scripts.run_sae_analysis", "--data-file", data, "--model", m, "--multi-layer"] + ex
            else:
                raise ValueError(task)
            jobs.append(dict(name=f"{task}_{m}_{ds}", model=m, cmd=cmd, done=done))
    return jobs


def is_done(job) -> bool:
    p = job.get("done")
    if not p or not Path(p).exists():
        return False
    if job.get("done_key"):          # removal: 파일이 있어도 중간에 끊겼을 수 있음 → 실행기가 이어서 처리
        return False
    return True


def detect_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.strip().splitlines()
        rows = [tuple(map(int, l.split(","))) for l in out]
        return [str(i) for i, _ in rows], min(m for _, m in rows) // 1024
    except Exception:
        return [], 0


def main():
    ap = argparse.ArgumentParser(description="여러 모델 동시 실행 (GPU당 작업 1개)")
    ap.add_argument("task", choices=["behavior", "sae", "removal"])
    ap.add_argument("--models", required=True, help="쉼표 목록 (run_one_experiment / run_removal의 모델 키)")
    ap.add_argument("--datasets", default="jobfair,lbox,mind", help="behavior·sae용 쉼표 목록")
    ap.add_argument("--gpus", default=None, help="사용할 GPU 번호 (기본: 전부)")
    ap.add_argument("--gpu-mem", type=int, default=None, help="GPU 한 장 메모리(GB). 기본: nvidia-smi로 감지")
    ap.add_argument("--extra", default="", help="각 작업에 덧붙일 인자 (따옴표로 묶어서)")
    ap.add_argument("--force", action="store_true", help="결과 파일이 있어도 다시 실행")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    found, mem = detect_gpus()
    gpus = a.gpus.split(",") if a.gpus else found
    mem = a.gpu_mem or mem or 40
    if not gpus and not a.dry_run:
        sys.exit("GPU를 찾지 못했습니다 (--gpus로 지정 가능)")

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    jobs = build_jobs(a.task, models, a.datasets.split(","), a.extra)
    todo = [j for j in jobs if a.force or not is_done(j)]
    skipped = [j["name"] for j in jobs if j not in todo]
    # 큰 모델(GPU 여러 장)을 먼저 배치해야 나중에 묶을 GPU가 남는다
    todo.sort(key=lambda j: -gpus_needed(j["model"], mem))

    print(f"GPU {gpus or '(dry-run)'} ({mem}GB/장), 작업 {len(todo)}개" + (f", 완료돼서 건너뜀 {len(skipped)}개" if skipped else ""))
    for j in todo:
        print(f"  [{gpus_needed(j['model'], mem)} GPU] {j['name']}: {' '.join(j['cmd'][1:])}")
    if a.dry_run or not todo:
        return

    LOG_DIR.mkdir(exist_ok=True)
    free, running, summary = list(gpus), [], {}
    t0 = time.time()
    while todo or running:
        # 끝난 작업 회수
        for r in running[:]:
            rc = r["proc"].poll()
            if rc is not None:
                running.remove(r)
                free.extend(r["gpus"])
                dt = (time.time() - r["start"]) / 60
                summary[r["job"]["name"]] = dict(returncode=rc, minutes=round(dt, 1), gpus=r["gpus"], log=str(r["log"]))
                print(f"{'완료' if rc == 0 else '실패'} {r['job']['name']} (GPU {','.join(r['gpus'])}, {dt:.0f}분)"
                      + ("" if rc == 0 else f" → 로그 확인: {r['log']}"), flush=True)
                json.dump(summary, open(LOG_DIR / f"parallel_{a.task}.json", "w"), ensure_ascii=False, indent=1)
        # 빈 GPU에 새 작업 배치
        for j in todo[:]:
            need = gpus_needed(j["model"], mem)
            if need > len(gpus):
                print(f"건너뜀 {j['name']}: GPU {need}장 필요"); todo.remove(j); continue
            if len(free) >= need:
                use = [free.pop(0) for _ in range(need)]
                log = LOG_DIR / f"{j['name']}.log"
                env = os.environ | {"CUDA_VISIBLE_DEVICES": ",".join(use), "PYTHONUNBUFFERED": "1",
                                    "TOKENIZERS_PARALLELISM": "false"}
                proc = subprocess.Popen(j["cmd"], cwd=PROJECT_ROOT, env=env,
                                        stdout=open(log, "w"), stderr=subprocess.STDOUT)
                running.append(dict(job=j, proc=proc, gpus=use, log=log, start=time.time()))
                todo.remove(j)
                print(f"시작 {j['name']} → GPU {','.join(use)}  (로그 {log.relative_to(PROJECT_ROOT)})", flush=True)
        time.sleep(5)

    ok = sum(v["returncode"] == 0 for v in summary.values())
    print(f"\n전체 {len(summary)}개 중 성공 {ok}개, {(time.time() - t0) / 60:.0f}분. 요약: logs/parallel_{a.task}.json")
    sys.exit(0 if ok == len(summary) else 1)


if __name__ == "__main__":
    main()
