"""
Simple 1:1 batch processing for pre-augmented datasets (JobFair, LBOX).

Each row in the CSV is already a complete prompt with identity injected.
This script just sends each row as-is to the API and collects the response.

No comparison multiplication. No ranking. Just: prompt → score (0-10).

Usage:
  # Prepare
  python -m scripts.run_one_batch prepare --data-file data/jobfair.csv --provider openai
  python -m scripts.run_one_batch prepare --data-file data/jobfair.csv --provider anthropic

  # Submit (sequential: one chunk at a time, wait for completion)
  python -m scripts.run_one_batch submit --provider openai
  python -m scripts.run_one_batch submit --provider anthropic

  # Check status
  python -m scripts.run_one_batch poll --provider openai

  # Cancel all + clean
  python -m scripts.run_one_batch cancel --provider openai
  python -m scripts.run_one_batch clean --provider openai
"""

import json, csv, argparse, os, sys, time, math
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BATCH_DIR = PROJECT_ROOT / "outputs" / "one_batch"

BATCH_CHUNK_SIZE = {
    "openai": 1000,
    "anthropic": 5000,
}

MODEL_REGISTRY = {
    "openai": {
        "gpt-4.1-nano":   {"id": "gpt-4.1-nano-2025-04-14"},
        "gpt-4.1-mini": {"id": "gpt-4.1-mini-2025-04-14"},
    },
    "anthropic": {
        "claude-sonnet-4.6": {"id": "claude-sonnet-4-6"},
        "claude-haiku-4.5":  {"id": "claude-haiku-4-5-20251001"},
    },
}

SYSTEM_PROMPT = (
    "Read the prompt carefully and respond with ONLY a single number from 0 to 10. "
    "Do not explain. Just the number."
)


# ═══════════════════════════════════════
# LOAD CSV
# ═══════════════════════════════════════

def load_csv(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ═══════════════════════════════════════
# PREPARE
# ═══════════════════════════════════════

def prepare(data_file, provider):
    """Create batch JSONL files from pre-augmented CSV."""
    rows = load_csv(data_file)
    dataset_name = Path(data_file).stem  # e.g. "jobfair" or "lbox"
    print(f"Loaded {len(rows)} rows from {data_file}")

    models = MODEL_REGISTRY.get(provider, {})
    if not models:
        print(f"No models for provider '{provider}'")
        return

    for mname, minfo in models.items():
        mid = minfo["id"]
        requests = []

        for i, row in enumerate(rows):
            prompt = row.get("Prompt", "").strip()
            if not prompt:
                continue

            req_id = f"{dataset_name}_{i:06d}"

            if provider == "openai":
                requests.append({
                    "custom_id": req_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": mid,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 5,
                        "temperature": 0,
                    },
                })
            elif provider == "anthropic":
                requests.append({
                    "custom_id": req_id,
                    "params": {
                        "model": mid,
                        "max_tokens": 5,
                        "temperature": 0,
                        "system": SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                })

        # Write chunks
        out_dir = BATCH_DIR / provider
        out_dir.mkdir(parents=True, exist_ok=True)

        chunk_size = BATCH_CHUNK_SIZE[provider]
        n_chunks = math.ceil(len(requests) / chunk_size)
        for ci in range(n_chunks):
            start = ci * chunk_size
            end = min(start + chunk_size, len(requests))
            chunk = requests[start:end]
            fname = out_dir / f"{dataset_name}_{mname}_chunk{ci:03d}.jsonl"
            with open(fname, "w") as f:
                for req in chunk:
                    f.write(json.dumps(req) + "\n")
            print(f"  {fname.name}: {len(chunk)} requests")

        print(f"  {mname}: {len(requests)} total -> {n_chunks} chunks (chunk_size={chunk_size})")


# ═══════════════════════════════════════
# SUBMIT (sequential: submit → wait → download → next)
# ═══════════════════════════════════════

def submit_openai():
    from openai import OpenAI
    client = OpenAI()
    batch_dir = BATCH_DIR / "openai"
    _sequential_submit_openai(client, batch_dir)


def submit_anthropic():
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic()
    batch_dir = BATCH_DIR / "anthropic"
    _sequential_submit_anthropic(client, batch_dir, Request, MessageCreateParamsNonStreaming)


def _sequential_submit_openai(client, batch_dir):
    jsonl_files = sorted(batch_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files. Run 'prepare' first.")
        return

    jobs = _load_jobs(batch_dir)
    done = {j["file"] for j in jobs if j.get("status") == "completed"}
    remaining = [f for f in jsonl_files if f.name not in done]
    print(f"{len(jsonl_files)} total, {len(done)} done, {len(remaining)} remaining\n")

    for i, jf in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {jf.name}")
        uploaded = client.files.create(file=open(jf, "rb"), purpose="batch")
        print(f"  Uploaded: {uploaded.id}")

        batch = None
        for attempt in range(10):
            try:
                batch = client.batches.create(
                    input_file_id=uploaded.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
                print(f"  Batch: {batch.id}")
                break
            except Exception as e:
                if "limit" in str(e).lower() or "429" in str(e):
                    wait = 60 * (attempt + 1)
                    print(f"  Rate limit. Waiting {wait}s ({attempt+1}/10)...")
                    time.sleep(wait)
                else:
                    raise
        if not batch:
            print("  SKIP: failed after 10 retries")
            continue

        # Poll
        while True:
            batch = client.batches.retrieve(batch.id)
            c = batch.request_counts
            print(f"  {batch.status} ({c.completed}/{c.total})", end="\r", flush=True)
            if batch.status in ("completed", "failed", "expired", "cancelled"):
                print()
                break
            time.sleep(30)

        rec = {"file": jf.name, "batch_id": batch.id, "status": batch.status}
        if batch.status == "completed" and batch.output_file_id:
            content = client.files.content(batch.output_file_id).text
            out = batch_dir / f"results_{jf.name}"
            with open(out, "w") as f:
                f.write(content)
            print(f"  Saved: {out.name}")
            rec["output_file"] = out.name
        else:
            print(f"  FAILED: {batch.status}")

        jobs.append(rec)
        _save_jobs(batch_dir, jobs)

    print(f"\nDone. {len(jobs)} jobs total.")


def _sequential_submit_anthropic(client, batch_dir, Request, MessageCreateParamsNonStreaming):
    jsonl_files = sorted(batch_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files. Run 'prepare' first.")
        return

    jobs = _load_jobs(batch_dir)
    done = {j["file"] for j in jobs if j.get("status") == "completed"}
    remaining = [f for f in jsonl_files if f.name not in done]
    print(f"{len(jsonl_files)} total, {len(done)} done, {len(remaining)} remaining\n")

    for i, jf in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {jf.name}")
        requests = []
        with open(jf) as f:
            for line in f:
                d = json.loads(line)
                requests.append(Request(
                    custom_id=d["custom_id"],
                    params=MessageCreateParamsNonStreaming(**d["params"]),
                ))

        batch = None
        for attempt in range(10):
            try:
                batch = client.messages.batches.create(requests=requests)
                print(f"  Batch: {batch.id}")
                break
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower() or "limit" in str(e).lower():
                    wait = 60 * (attempt + 1)
                    print(f"  Rate limit. Waiting {wait}s ({attempt+1}/10)...")
                    time.sleep(wait)
                else:
                    raise
        if not batch:
            print("  SKIP: failed after 10 retries")
            continue

        # Poll
        while True:
            batch = client.messages.batches.retrieve(batch.id)
            c = batch.request_counts
            d = c.succeeded + c.errored + c.canceled + c.expired
            print(f"  {batch.processing_status} ({d}/{d + c.processing})", end="\r", flush=True)
            if batch.processing_status == "ended":
                print()
                break
            time.sleep(30)

        out = batch_dir / f"results_{jf.name}"
        with open(out, "w") as f:
            for result in client.messages.batches.results(batch.id):
                f.write(json.dumps({
                    "custom_id": result.custom_id,
                    "result": result.result.model_dump(),
                }) + "\n")
        print(f"  Saved: {out.name} ({c.succeeded} ok, {c.errored} err)")

        jobs.append({"file": jf.name, "batch_id": batch.id, "status": "completed", "output_file": out.name})
        _save_jobs(batch_dir, jobs)

    print(f"\nDone. {len(jobs)} jobs total.")


# ═══════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════

def _load_jobs(d):
    f = d / "jobs.json"
    return json.load(open(f)) if f.exists() else []

def _save_jobs(d, jobs):
    with open(d / "jobs.json", "w") as f:
        json.dump(jobs, f, indent=2)


# ═══════════════════════════════════════
# CANCEL / CLEAN / POLL
# ═══════════════════════════════════════

def cancel(provider):
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        n = 0
        for b in client.batches.list(limit=100):
            if b.status in ("validating", "in_progress", "finalizing"):
                try:
                    client.batches.cancel(b.id)
                    print(f"  Cancelled: {b.id}")
                    n += 1
                except Exception as e:
                    print(f"  Failed: {b.id} — {e}")
        print(f"Cancelled {n}")

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        n = 0
        for b in client.messages.batches.list(limit=100):
            if b.processing_status == "in_progress":
                try:
                    client.messages.batches.cancel(b.id)
                    print(f"  Cancelled: {b.id}")
                    n += 1
                except Exception as e:
                    print(f"  Failed: {b.id} — {e}")
        print(f"Cancelled {n}")


def clean(provider):
    d = BATCH_DIR / provider
    if not d.exists():
        print(f"  {d}: not found")
        return
    n = 0
    for f in d.glob("*"):
        f.unlink()
        n += 1
    print(f"  Removed {n} files from {d}")


def poll(provider):
    jobs = _load_jobs(BATCH_DIR / provider)
    if not jobs:
        print("No jobs found.")
        return
    for j in jobs:
        print(f"  {j['file']:50s} {j.get('status', '?')}")
    done = sum(1 for j in jobs if j.get("status") == "completed")
    print(f"\n{done}/{len(jobs)} completed")


# ═══════════════════════════════════════
# CLI
# ═══════════════════════════════════════

def _check_key(provider):
    keys = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    k = keys.get(provider)
    if k and not os.environ.get(k):
        print(f"Error: {k} not set")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="Simple 1:1 batch (JobFair/LBOX)")
    p.add_argument("action", choices=["prepare", "submit", "poll", "cancel", "clean"])
    p.add_argument("--data-file", type=str, default=None)
    p.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    args = p.parse_args()

    if args.action == "prepare":
        if not args.data_file:
            print("Error: --data-file required")
            sys.exit(1)
        if not Path(args.data_file).exists():
            print(f"Error: {args.data_file} not found")
            sys.exit(1)
        prepare(args.data_file, args.provider)

    elif args.action == "submit":
        _check_key(args.provider)
        {"openai": submit_openai, "anthropic": submit_anthropic}[args.provider]()

    elif args.action == "poll":
        poll(args.provider)

    elif args.action == "cancel":
        _check_key(args.provider)
        cancel(args.provider)

    elif args.action == "clean":
        clean(args.provider)


if __name__ == "__main__":
    main()