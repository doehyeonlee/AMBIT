"""
Local inference for pre-augmented datasets (JobFair, LBOX, WinoIdentity).

Each row in the CSV is already a complete prompt.
Sends each row to a local HuggingFace model and collects the response.

Usage:
  python -m scripts.run_one_experiment --data-file data/jobfair.csv --model gemma-2-9b
  python -m scripts.run_one_experiment --data-file data/lbox.csv --model gemma-3-4b
  python -m scripts.run_one_experiment --list-models
"""

import json, csv, argparse, os, sys, time, re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "local"

MODEL_REGISTRY = {
    "gemma-3-4b":   {"id": "google/gemma-3-4b-it",   "sae": "gemmascope2"},
    "gemma-2-9b":   {"id": "google/gemma-2-9b-it",   "sae": "gemmascope"},
    "qwen3.5-4b":   {"id": "Qwen/Qwen3.5-4B",       "sae": None},
    "mistral-7b":   {"id": "mistralai/Mistral-7B-Instruct-v0.3", "sae": None},
}

SYSTEM_PROMPT = (
    "Read the prompt carefully and respond with ONLY a single number from 0 to 10. "
    "Do not explain. Just the number."
)

# Loaded model cache
_model_cache = {}


def load_model(model_id):
    """Load HF model and tokenizer, cached. Uses model-specific loaders."""
    if model_id in _model_cache:
        return _model_cache[model_id]

    import torch

    if "pythia" in model_id.lower():
        # Pythia: use GPTNeoXForCausalLM (not AutoModelForCausalLM)
        from transformers import GPTNeoXForCausalLM, AutoTokenizer
        print(f"Loading {model_id} (GPTNeoX)...")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = GPTNeoXForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()

    elif "ministral" in model_id.lower() or "mistral" in model_id.lower():
        # Mistral: use dedicated loader with FP8 quantization
        from transformers import AutoTokenizer
        print(f"Loading {model_id} (Mistral)...")
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        try:
            from transformers import Mistral3ForConditionalGeneration, FineGrainedFP8Config
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id, device_map="auto",
                quantization_config=FineGrainedFP8Config(dequantize=True))
        except (ImportError, Exception) as e:
            print(f"  Mistral3/FP8 failed ({e}), falling back to AutoModelForCausalLM...")
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model.eval()

    else:
        # Default: Gemma and other models
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"Loading {model_id}...")
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model.eval()

    print(f"  Loaded on {next(model.parameters()).device}")
    _model_cache[model_id] = (tok, model)
    return tok, model


def generate(tok, model, prompt, system=SYSTEM_PROMPT, max_tokens=20):
    """Generate response from local model."""
    import torch

    # Build messages — try system role, fall back to merging
    messages_with_sys = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    messages_no_sys = [{"role": "user", "content": f"{system}\n\n{prompt}"}]

    input_text = None
    if hasattr(tok, "apply_chat_template"):
        try:
            input_text = tok.apply_chat_template(messages_with_sys, tokenize=False, add_generation_prompt=True)
        except Exception:
            input_text = tok.apply_chat_template(messages_no_sys, tokenize=False, add_generation_prompt=True)
    if not input_text:
        input_text = f"{system}\n\n{prompt}"

    inputs = tok(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=0.01, do_sample=False,
        )
    response = tok.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response.strip()


def load_csv(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_score(raw):
    """Extract 0-10 score from response."""
    if not raw:
        return None
    m = re.search(r"\b(\d{1,2})\b", raw)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 10:
            return val
    return None


def parse_choice(raw):
    """Extract A/B choice from response."""
    if not raw:
        return None
    for ch in raw.strip().upper():
        if ch in "AB":
            return ch
    return None


def run(data_file, model_name, max_probes=None, output_dir=None):
    """Run inference on all rows."""
    if model_name not in MODEL_REGISTRY:
        print(f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    model_id = MODEL_REGISTRY[model_name]["id"]
    dataset_name = Path(data_file).stem

    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(data_file)
    if max_probes:
        rows = rows[:max_probes]
    print(f"Loaded {len(rows)} rows from {data_file}")

    tok, model = load_model(model_id)

    # Detect task type from data
    is_coref = any(r.get("winobias_task_type", "").startswith("type") for r in rows[:10])
    is_jobfair = any("jobfair" in r.get("winobias_task_type", "") for r in rows[:10])
    task = "coref" if is_coref else "score"
    max_tokens = 10 if task == "coref" else 5
    print(f"Task type: {task} (max_tokens={max_tokens})")

    results = []
    t_start = time.time()

    for i, row in enumerate(rows):
        prompt = row.get("Prompt", "").strip()
        if not prompt:
            continue

        try:
            raw = generate(tok, model, prompt, SYSTEM_PROMPT, max_tokens)
        except Exception as e:
            raw = None
            if i < 3:
                print(f"  Error at {i}: {e}")

        # Parse based on task
        if task == "coref":
            parsed = parse_choice(raw)
            correct = None
            if parsed:
                occs_str = row.get("occs_in_samples", "[]")
                occs = [s.strip().strip("'\"") for s in occs_str.strip("[]").split(",") if s.strip()]
                ref = row.get("referent_occ", "")
                if ref in occs:
                    ref_idx = occs.index(ref)
                    correct = (parsed == chr(65 + ref_idx))
            result = {"chosen": parsed, "correct": correct}
        else:
            score = parse_score(raw)
            result = {"score": score}

        # Common fields
        demo_str = row.get("demographic_identifier", "[]")
        demo = [s.strip().strip("'\"") for s in demo_str.strip("[]").split(",") if s.strip()]

        result.update({
            "row_idx": i,
            "raw_answer": raw,
            "demographic": demo,
            "referent_occ": row.get("referent_occ", ""),
            "stereotype_label": row.get("stereotype_label", ""),
            "task_type": row.get("winobias_task_type", ""),
        })
        results.append(result)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(rows) - i - 1) / rate if rate > 0 else 0
            if task == "coref":
                answered = sum(1 for r in results if r.get("chosen"))
                print(f"  {i+1}/{len(rows)} ({answered} answered, {rate:.1f}/s, ETA {eta/60:.0f}m)")
            else:
                scored = sum(1 for r in results if r.get("score") is not None)
                print(f"  {i+1}/{len(rows)} ({scored} scored, {rate:.1f}/s, ETA {eta/60:.0f}m)")

    # Save raw results
    raw_file = out_dir / f"raw_{dataset_name}_{model_name}.json"
    with open(raw_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Compute & save metrics
    metrics = _compute_metrics(results, task)
    metrics_file = out_dir / f"metrics_{dataset_name}_{model_name}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f}m. Saved: {raw_file.name}, {metrics_file.name}")
    _print_summary(metrics, task, model_name)


def _compute_metrics(results, task):
    """Group by demographic and compute metrics."""
    by_demo = defaultdict(list)
    for r in results:
        demo = r.get("demographic", [])
        key = "+".join(sorted(demo)) if demo else "(baseline)"
        by_demo[key].append(r)

    per_identity = {}
    for key, group in by_demo.items():
        parts = key.split("+") if key != "(baseline)" else []
        m = {"n": len(group)}

        if task == "coref":
            valid = [r for r in group if r.get("correct") is not None]
            m["accuracy"] = sum(1 for r in valid if r["correct"]) / len(valid) if valid else None
        else:
            scores = [r["score"] for r in group if r.get("score") is not None]
            m["mean_score"] = sum(scores) / len(scores) if scores else None
            m["n_scored"] = len(scores)

        m["identity_type"] = "baseline" if not parts else ("single" if len(parts) == 1 else "multi")
        per_identity[key] = m

    # Overall
    overall = {"total": len(results), "n_identities": len(per_identity)}

    if task == "coref":
        all_valid = [r for r in results if r.get("correct") is not None]
        overall["accuracy"] = sum(1 for r in all_valid if r["correct"]) / len(all_valid) if all_valid else None
        accs = {k: v["accuracy"] for k, v in per_identity.items() if v.get("accuracy") is not None}
        if len(accs) >= 2:
            overall["disparity"] = max(accs.values()) - min(accs.values())
            overall["best"] = max(accs, key=accs.get)
            overall["worst"] = min(accs, key=accs.get)
    else:
        all_scores = [r["score"] for r in results if r.get("score") is not None]
        overall["mean_score"] = sum(all_scores) / len(all_scores) if all_scores else None
        overall["n_scored"] = len(all_scores)
        means = {k: v["mean_score"] for k, v in per_identity.items() if v.get("mean_score") is not None}
        if len(means) >= 2:
            overall["disparity"] = max(means.values()) - min(means.values())
            overall["highest_scored"] = max(means, key=means.get)
            overall["lowest_scored"] = min(means, key=means.get)

    # Interaction terms for multi-identity
    base_val = per_identity.get("(baseline)", {}).get("accuracy" if task == "coref" else "mean_score")
    i_terms = []
    for key, m in per_identity.items():
        if m["identity_type"] != "multi":
            continue
        parts = key.split("+")
        if len(parts) != 2:
            continue
        val_multi = m.get("accuracy" if task == "coref" else "mean_score")
        val_a = per_identity.get(parts[0], {}).get("accuracy" if task == "coref" else "mean_score")
        val_b = per_identity.get(parts[1], {}).get("accuracy" if task == "coref" else "mean_score")
        if all(v is not None for v in [val_multi, val_a, val_b, base_val]):
            I = val_multi - val_a - val_b + base_val
            m["I_term"] = I
            i_terms.append(I)

    if i_terms:
        import numpy as np
        overall["mean_I"] = float(np.mean(i_terms))
        overall["n_negative_I"] = sum(1 for v in i_terms if v < 0)
        overall["n_I"] = len(i_terms)

    return {"per_identity": per_identity, "overall": overall}


def _print_summary(metrics, task, model_name):
    ov = metrics["overall"]
    print(f"\n  --- {model_name} ---")
    if task == "coref":
        print(f"  Accuracy: {ov.get('accuracy')}")
    else:
        print(f"  Mean score: {ov.get('mean_score')}")
        print(f"  Scored: {ov.get('n_scored')}/{ov.get('total')}")
    if ov.get("disparity") is not None:
        print(f"  Disparity: {ov['disparity']:.3f}")
        if task == "coref":
            print(f"    Best:  {ov.get('best')}")
            print(f"    Worst: {ov.get('worst')}")
        else:
            print(f"    Highest: {ov.get('highest_scored')}")
            print(f"    Lowest:  {ov.get('lowest_scored')}")
    if ov.get("mean_I") is not None:
        print(f"  Mean interaction: {ov['mean_I']:.4f} ({ov['n_negative_I']}/{ov['n_I']} negative)")


def main():
    p = argparse.ArgumentParser(description="Local inference for pre-augmented datasets")
    p.add_argument("--data-file", type=str, help="Path to CSV (jobfair.csv, lbox.csv, winoidentity.csv)")
    p.add_argument("--model", type=str, help="Model name (gemma-2-9b, gemma-3-4b)")
    p.add_argument("--max-probes", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--list-models", action="store_true")
    args = p.parse_args()

    if args.list_models:
        print("Available models:")
        for name, info in MODEL_REGISTRY.items():
            sae = info.get("sae", "—")
            print(f"  {name:15s} -> {info['id']:30s} SAE={sae}")
        return

    if not args.data_file or not args.model:
        print("Error: --data-file and --model required")
        print("  python -m scripts.run_one_experiment --data-file data/jobfair.csv --model gemma-2-9b")
        sys.exit(1)

    if not Path(args.data_file).exists():
        print(f"Error: {args.data_file} not found")
        sys.exit(1)

    run(args.data_file, args.model, args.max_probes, args.output_dir)


if __name__ == "__main__":
    main()