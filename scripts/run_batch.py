"""
Batch processing for API providers (OpenAI, Anthropic, Google).

Prepares JSONL batch files, submits them, polls for completion, and
downloads results into the same format as run_experiment.py outputs.

Usage:
  # Step 1: Prepare batch files from experiment config
  python -m scripts.run_batch prepare --context coref --data-file data/winoidentity.csv --provider openai
  python -m scripts.run_batch prepare --context coref --data-file data/winoidentity.csv --provider anthropic
  python -m scripts.run_batch prepare --context coref --data-file data/winoidentity.csv --provider google

  # Step 2: Submit batches
  python -m scripts.run_batch submit --provider openai
  python -m scripts.run_batch submit --provider anthropic
  python -m scripts.run_batch submit --provider google

  # Step 3: Poll and download results
  python -m scripts.run_batch poll --provider openai
  python -m scripts.run_batch poll --provider anthropic
  python -m scripts.run_batch poll --provider google
"""

import json, argparse, os, sys, time, math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_experiment import (
    load_csv, format_coref_probe, build_ranking_comparisons,
    COREF_SYSTEM, RANKING_TEMPLATES, COMPARISON_CONFIGS,
    MODEL_REGISTRY, parse_choice, parse_ranking, detect_refusal,
    compute_coref_metrics, compute_ranking_metrics,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BATCH_DIR = PROJECT_ROOT / "outputs" / "batch"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "behavioral"

BATCH_CHUNK_SIZE = 5000  # Requests per batch file


# ═══════════════════════════════════════
# PREPARE BATCH FILES
# ═══════════════════════════════════════

def prepare_coref_batch(data_file, provider):
    """Prepare batch JSONL files for coreference task."""
    probes = load_csv(data_file)
    print(f"Loaded {len(probes)} probes for coref")

    models = [m for p, ms in MODEL_REGISTRY.items() if p == provider for m in ms.values()]
    if not models:
        print(f"No models for provider {provider}")
        return

    for model_info in models:
        mid = model_info["id"]
        mname = next(k for p, ms in MODEL_REGISTRY.items() if p == provider for k, v in ms.items() if v["id"] == mid)

        requests = []
        for i, p in enumerate(probes):
            prompt_text = format_coref_probe(p)
            if not prompt_text:
                continue
            req_id = f"coref_{i:06d}"
            requests.append(_make_request(provider, mid, req_id, prompt_text, COREF_SYSTEM, 10))

        _write_batch_files(provider, f"coref_{mname}", requests)


def prepare_ranking_batch(context, data_file, provider):
    """Prepare batch JSONL files for ranking task (job/legal)."""
    probes = load_csv(data_file)
    comparisons = build_ranking_comparisons()
    template = RANKING_TEMPLATES[context]
    print(f"Loaded {len(probes)} probes, {len(comparisons)} comparisons for {context}")

    models = [m for p, ms in MODEL_REGISTRY.items() if p == provider for m in ms.values()]
    if not models:
        print(f"No models for provider {provider}")
        return

    import random
    random.seed(42)

    for model_info in models:
        mid = model_info["id"]
        mname = next(k for p, ms in MODEL_REGISTRY.items() if p == provider for k, v in ms.items() if v["id"] == mid)

        requests = []
        idx = 0
        for comp in comparisons:
            slots = comp["slots"]
            for probe in probes:
                shuffled = list(slots)
                random.shuffle(shuffled)
                for si, s in enumerate(shuffled):
                    s["label"] = chr(65 + si)

                prompt_text = template["build"](probe, shuffled)
                req_id = f"{context}_{idx:06d}"
                meta = json.dumps({
                    "comp_name": comp["comp_name"], "comp_type": comp["comp_type"],
                    "gender": comp["gender"],
                    "slots": [dict(s) for s in shuffled],
                })
                requests.append(_make_request(provider, mid, req_id, prompt_text,
                                              template["system"], 30, meta))
                idx += 1

        _write_batch_files(provider, f"{context}_{mname}", requests)


def _make_request(provider, model_id, custom_id, prompt, system, max_tokens, metadata=None):
    """Create a single batch request in the format required by each provider."""
    if provider == "openai":
        body = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        req = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        if metadata:
            req["metadata"] = metadata
        return req

    elif provider == "anthropic":
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        return {
            "custom_id": custom_id,
            "params": {
                "model": model_id,
                "max_tokens": max_tokens,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            "metadata": metadata,
        }

    elif provider == "google":
        req = {
            "key": custom_id,
            "request": {
                "contents": [{"parts": [{"text": prompt}], "role": "user"}],
                "system_instruction": {"parts": [{"text": system}]},
                "generation_config": {"temperature": 0, "max_output_tokens": max_tokens},
            },
        }
        if metadata:
            req["metadata"] = metadata
        return req


def _write_batch_files(provider, prefix, requests):
    """Write batch JSONL files, splitting into BATCH_CHUNK_SIZE per file for all providers."""
    out_dir = BATCH_DIR / provider
    out_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = math.ceil(len(requests) / BATCH_CHUNK_SIZE)
    for chunk_idx in range(n_chunks):
        start = chunk_idx * BATCH_CHUNK_SIZE
        end = min(start + BATCH_CHUNK_SIZE, len(requests))
        chunk = requests[start:end]
        fname = out_dir / f"{prefix}_chunk{chunk_idx:03d}.jsonl"
        with open(fname, "w") as f:
            for req in chunk:
                f.write(json.dumps(req) + "\n")
        print(f"  Wrote {len(chunk)} requests to {fname}")

    print(f"  Total: {len(requests)} requests -> {n_chunks} chunks for {prefix}")


# ═══════════════════════════════════════
# SEQUENTIAL SUBMIT (submit → wait → download → next chunk)
# ═══════════════════════════════════════

def submit_openai():
    """Submit OpenAI batches sequentially: one chunk at a time, wait for completion."""
    from openai import OpenAI
    client = OpenAI()
    batch_dir = BATCH_DIR / "openai"
    jsonl_files = sorted(batch_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found. Run 'prepare' first.")
        return

    jobs = _load_jobs(batch_dir)
    done_files = {j["file"] for j in jobs if j.get("status") == "completed"}
    remaining = [f for f in jsonl_files if f.name not in done_files]
    print(f"Total: {len(jsonl_files)} files, {len(done_files)} already done, {len(remaining)} remaining\n")

    for i, jsonl_file in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {jsonl_file.name}")

        # Upload
        print(f"  Uploading...", end=" ", flush=True)
        uploaded = client.files.create(file=open(jsonl_file, "rb"), purpose="batch")
        print(f"file={uploaded.id}")

        # Submit
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"  Submitted: {batch.id}")

        # Wait for completion
        while True:
            batch = client.batches.retrieve(batch.id)
            status = batch.status
            counts = batch.request_counts
            print(f"  Status: {status} (done={counts.completed}/{counts.total})", end="\r", flush=True)
            if status in ("completed", "failed", "expired", "cancelled"):
                print()
                break
            time.sleep(30)

        # Download results
        job_record = {"file": jsonl_file.name, "batch_id": batch.id, "status": status}
        if status == "completed" and batch.output_file_id:
            content = client.files.content(batch.output_file_id).text
            out_file = batch_dir / f"results_{jsonl_file.name}"
            with open(out_file, "w") as f:
                f.write(content)
            print(f"  Saved: {out_file.name}")
            job_record["output_file"] = out_file.name
        else:
            print(f"  FAILED: {status}")

        jobs.append(job_record)
        _save_jobs(batch_dir, jobs)

    print(f"\nAll done. {len(jobs)} total jobs.")


def submit_anthropic():
    """Submit Anthropic batches sequentially."""
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    batch_dir = BATCH_DIR / "anthropic"
    jsonl_files = sorted(batch_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found. Run 'prepare' first.")
        return

    jobs = _load_jobs(batch_dir)
    done_files = {j["file"] for j in jobs if j.get("status") == "completed"}
    remaining = [f for f in jsonl_files if f.name not in done_files]
    print(f"Total: {len(jsonl_files)} files, {len(done_files)} already done, {len(remaining)} remaining\n")

    for i, jsonl_file in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {jsonl_file.name}")

        # Build requests
        requests = []
        with open(jsonl_file) as f:
            for line in f:
                data = json.loads(line)
                requests.append(Request(
                    custom_id=data["custom_id"],
                    params=MessageCreateParamsNonStreaming(**data["params"]),
                ))

        # Submit
        print(f"  Submitting {len(requests)} requests...", end=" ", flush=True)
        batch = client.messages.batches.create(requests=requests)
        print(f"batch={batch.id}")

        # Wait
        while True:
            batch = client.messages.batches.retrieve(batch.id)
            counts = batch.request_counts
            done = counts.succeeded + counts.errored + counts.canceled + counts.expired
            total = done + counts.processing
            print(f"  Status: {batch.processing_status} ({done}/{total} done)", end="\r", flush=True)
            if batch.processing_status == "ended":
                print()
                break
            time.sleep(30)

        # Download
        out_file = batch_dir / f"results_{jsonl_file.name}"
        with open(out_file, "w") as f:
            for result in client.messages.batches.results(batch.id):
                f.write(json.dumps({
                    "custom_id": result.custom_id,
                    "result": result.result.model_dump(),
                }) + "\n")
        print(f"  Saved: {out_file.name} ({counts.succeeded} succeeded, {counts.errored} errored)")

        job_record = {"file": jsonl_file.name, "batch_id": batch.id, "status": "completed",
                      "output_file": out_file.name}
        jobs.append(job_record)
        _save_jobs(batch_dir, jobs)

    print(f"\nAll done. {len(jobs)} total jobs.")


def submit_google():
    """Submit Google batches sequentially."""
    from google import genai
    from google.genai import types
    client = genai.Client()
    batch_dir = BATCH_DIR / "google"
    jsonl_files = sorted(batch_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found. Run 'prepare' first.")
        return

    jobs = _load_jobs(batch_dir)
    done_files = {j["file"] for j in jobs if j.get("status") == "completed"}
    remaining = [f for f in jsonl_files if f.name not in done_files]
    print(f"Total: {len(jsonl_files)} files, {len(done_files)} already done, {len(remaining)} remaining\n")

    completed_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

    for i, jsonl_file in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {jsonl_file.name}")

        # Determine model from filename
        model_id = None
        for mname, info in MODEL_REGISTRY.get("google", {}).items():
            if mname.replace("-", "") in jsonl_file.stem.replace("-", ""):
                model_id = info["id"]
                break
        if not model_id:
            model_id = "gemini-3-flash-preview"

        # Upload
        print(f"  Uploading for {model_id}...", end=" ", flush=True)
        uploaded = client.files.upload(
            file=str(jsonl_file),
            config=types.UploadFileConfig(display_name=jsonl_file.stem, mime_type="jsonl"),
        )
        print(f"file={uploaded.name}")

        # Submit
        batch = client.batches.create(
            model=model_id, src=uploaded.name,
            config={"display_name": jsonl_file.stem},
        )
        print(f"  Submitted: {batch.name}")

        # Wait
        while True:
            batch = client.batches.get(name=batch.name)
            print(f"  Status: {batch.state.name}", end="\r", flush=True)
            if batch.state.name in completed_states:
                print()
                break
            time.sleep(30)

        # Download
        job_record = {"file": jsonl_file.name, "batch_name": batch.name, "model": model_id, "status": batch.state.name}
        if batch.state.name == "JOB_STATE_SUCCEEDED":
            out_file = batch_dir / f"results_{jsonl_file.name}"
            if batch.dest and batch.dest.file_name:
                content = client.files.download(file=batch.dest.file_name)
                with open(out_file, "wb") as f:
                    f.write(content)
            elif batch.dest and batch.dest.inlined_responses:
                with open(out_file, "w") as f:
                    for ri, resp in enumerate(batch.dest.inlined_responses):
                        text = resp.response.text if resp.response else None
                        f.write(json.dumps({"key": f"req_{ri}", "text": text}) + "\n")
            print(f"  Saved: {out_file.name}")
            job_record["output_file"] = out_file.name
            job_record["status"] = "completed"
        else:
            print(f"  FAILED: {batch.state.name}")

        jobs.append(job_record)
        _save_jobs(batch_dir, jobs)

    print(f"\nAll done. {len(jobs)} total jobs.")


def _load_jobs(batch_dir):
    jobs_file = batch_dir / "jobs.json"
    if jobs_file.exists():
        return json.load(open(jobs_file))
    return []


def _save_jobs(batch_dir, jobs):
    with open(batch_dir / "jobs.json", "w") as f:
        json.dump(jobs, f, indent=2)


# ═══════════════════════════════════════
# POLL (legacy — submit now does poll inline, but keep for checking status)
# ═══════════════════════════════════════

def poll_openai():
    """Check status of OpenAI jobs."""
    jobs = _load_jobs(BATCH_DIR / "openai")
    if not jobs:
        print("No jobs found.")
        return
    for j in jobs:
        status = j.get("status", "unknown")
        print(f"  {j['file']:40s} {status}")
    done = sum(1 for j in jobs if j.get("status") == "completed")
    print(f"\n{done}/{len(jobs)} completed")


def poll_anthropic():
    """Check status of Anthropic jobs."""
    jobs = _load_jobs(BATCH_DIR / "anthropic")
    if not jobs:
        print("No jobs found.")
        return
    for j in jobs:
        status = j.get("status", "unknown")
        print(f"  {j['file']:40s} {status}")
    done = sum(1 for j in jobs if j.get("status") == "completed")
    print(f"\n{done}/{len(jobs)} completed")


def poll_google():
    """Check status of Google jobs."""
    jobs = _load_jobs(BATCH_DIR / "google")
    if not jobs:
        print("No jobs found.")
        return
    for j in jobs:
        status = j.get("status", "unknown")
        print(f"  {j['file']:40s} {status}")
    done = sum(1 for j in jobs if j.get("status") == "completed")
    print(f"\n{done}/{len(jobs)} completed")


# ═══════════════════════════════════════
# CLI
# ═══════════════════════════════════════

# ═══════════════════════════════════════
# CANCEL ALL BATCHES
# ═══════════════════════════════════════

def cancel_openai():
    """Cancel all in-progress OpenAI batches."""
    from openai import OpenAI
    client = OpenAI()
    print("Cancelling all OpenAI batches...")
    cancelled = 0
    for batch in client.batches.list(limit=100):
        if batch.status in ("validating", "in_progress", "finalizing"):
            try:
                client.batches.cancel(batch.id)
                print(f"  Cancelled: {batch.id} (was {batch.status})")
                cancelled += 1
            except Exception as e:
                print(f"  Failed to cancel {batch.id}: {e}")
    print(f"Cancelled {cancelled} batches")


def cancel_anthropic():
    """Cancel all in-progress Anthropic batches."""
    import anthropic
    client = anthropic.Anthropic()
    print("Cancelling all Anthropic batches...")
    cancelled = 0
    for batch in client.messages.batches.list(limit=100):
        if batch.processing_status == "in_progress":
            try:
                client.messages.batches.cancel(batch.id)
                print(f"  Cancelled: {batch.id}")
                cancelled += 1
            except Exception as e:
                print(f"  Failed to cancel {batch.id}: {e}")
    print(f"Cancelled {cancelled} batches")


def cancel_google():
    """Cancel all in-progress Google batches."""
    from google import genai
    client = genai.Client()
    jobs_file = BATCH_DIR / "google" / "jobs.json"
    if not jobs_file.exists():
        print("No jobs.json found for Google.")
        return
    jobs = json.load(open(jobs_file))
    cancelled = 0
    for job in jobs:
        bname = job.get("batch_name")
        if not bname:
            continue
        try:
            batch = client.batches.get(name=bname)
            if batch.state.name in ("JOB_STATE_PENDING", "JOB_STATE_RUNNING"):
                client.batches.cancel(name=bname)
                print(f"  Cancelled: {bname}")
                cancelled += 1
        except Exception as e:
            print(f"  Failed to cancel {bname}: {e}")
    print(f"Cancelled {cancelled} batches")


def clean_batch_files(provider=None):
    """Remove all batch JSONL files and jobs.json."""
    providers = [provider] if provider else ["openai", "anthropic", "google"]
    for prov in providers:
        prov_dir = BATCH_DIR / prov
        if not prov_dir.exists():
            continue
        removed = 0
        for f in prov_dir.glob("*"):
            f.unlink()
            removed += 1
        if removed:
            print(f"  Removed {removed} files from {prov_dir}")
        else:
            print(f"  {prov_dir}: already clean")


# ═══════════════════════════════════════
# CLI
# ═══════════════════════════════════════

def _check_api_key(provider):
    """Check if API key is set for provider."""
    env_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}
    key = env_map.get(provider, "")
    if key and not os.environ.get(key):
        print(f"Error: {key} not set. Run: export {key}=...")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="CCIP Batch Processing")
    p.add_argument("action", choices=["prepare", "submit", "poll", "cancel", "clean"],
                   help="prepare=create JSONL, submit=upload & start, poll=wait & download, "
                        "cancel=stop all running batches, clean=delete all batch files")
    p.add_argument("--context", choices=["coref", "job", "legal"])
    p.add_argument("--data-file", type=str, default=None)
    p.add_argument("--provider", choices=["openai", "anthropic", "google"], required=True)
    p.add_argument("--max-probes", type=int, default=None)
    args = p.parse_args()

    if args.action == "prepare":
        if not args.context:
            print("Error: --context required for prepare")
            sys.exit(1)
        data_file = args.data_file
        if not data_file:
            defaults = {"coref": "winoidentity.csv", "job": "jobfair.csv", "legal": "lbox.csv"}
            data_file = str(DATA_DIR / defaults[args.context])
        if not Path(data_file).exists():
            print(f"Error: {data_file} not found")
            sys.exit(1)

        if args.context == "coref":
            prepare_coref_batch(data_file, args.provider)
        else:
            prepare_ranking_batch(args.context, data_file, args.provider)

    elif args.action == "submit":
        _check_api_key(args.provider)
        {"openai": submit_openai, "anthropic": submit_anthropic, "google": submit_google}[args.provider]()

    elif args.action == "poll":
        _check_api_key(args.provider)
        {"openai": poll_openai, "anthropic": poll_anthropic, "google": poll_google}[args.provider]()

    elif args.action == "cancel":
        _check_api_key(args.provider)
        {"openai": cancel_openai, "anthropic": cancel_anthropic, "google": cancel_google}[args.provider]()

    elif args.action == "clean":
        clean_batch_files(args.provider)


if __name__ == "__main__":
    main()