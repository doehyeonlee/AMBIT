"""
Parse and analyze experiment results from all sources.

Sources:
  1. API Batch results (run_one_batch.py)  → outputs/one_batch/{openai,anthropic}/results_*.jsonl
  2. Local model results (run_one_experiment.py) → outputs/local/{dataset}/raw_*.json
  3. SAE analysis (run_sae_analysis.py) → outputs/sae/*/sae_results.json

Output:
  - Consolidated CSV per dataset with all models' scores
  - Summary statistics by identity
  - Key findings report

Usage:
  python -m scripts.analyze_results --dataset jobfair
  python -m scripts.analyze_results --dataset lbox
  python -m scripts.analyze_results --dataset mind
  python -m scripts.analyze_results --dataset winoidentity
  python -m scripts.analyze_results --all
"""

import json, csv, argparse, sys, re, os
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"


# ═══════════════════════════════════════
# PARSE BATCH RESULTS
# ═══════════════════════════════════════

def parse_openai_results(results_file):
    """Parse OpenAI batch results JSONL."""
    records = []
    with open(results_file) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            custom_id = data.get("custom_id", "")
            resp = data.get("response", {})
            body = resp.get("body", {})
            choices = body.get("choices", [])
            text = choices[0]["message"]["content"].strip() if choices else None
            score = _parse_score(text)
            records.append({"custom_id": custom_id, "raw": text, "score": score})
    return records


def parse_anthropic_results(results_file):
    """Parse Anthropic batch results JSONL."""
    records = []
    with open(results_file) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            custom_id = data.get("custom_id", "")
            result = data.get("result", {})
            if result.get("type") == "succeeded":
                msg = result.get("message", {})
                content = msg.get("content", [])
                text = content[0]["text"].strip() if content else None
            else:
                text = None
            score = _parse_score(text)
            records.append({"custom_id": custom_id, "raw": text, "score": score})
    return records


def parse_local_results(results_file):
    """Parse local model results JSON. Handles both active and legacy formats."""
    with open(results_file) as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, dict):
        # Legacy format might be {comp_key: [results]} or {field: value}
        # Try to flatten nested lists
        flat = []
        for k, v in data.items():
            if isinstance(v, list):
                flat.extend(v)
            elif isinstance(v, dict):
                flat.append(v)
        if not flat:
            print(f"    Skipping {Path(results_file).name}: unrecognized dict format")
            return []
        data = flat

    if not isinstance(data, list):
        print(f"    Skipping {Path(results_file).name}: not a list (type={type(data).__name__})")
        return []

    records = []
    recovered = 0
    for i, r in enumerate(data):
        if isinstance(r, str):
            # Legacy format: list of raw answer strings
            score = _parse_score(r)
            records.append({
                "custom_id": f"row_{i:06d}",
                "raw": r,
                "score": score,
                "demographic": [],
                "referent_occ": "",
            })
            continue

        if not isinstance(r, dict):
            continue

        original_score = r.get("score")
        raw = r.get("raw_answer") or r.get("answer") or r.get("raw") or r.get("text")
        reparsed = _parse_score(raw) if raw else None
        score = original_score if original_score is not None else reparsed
        if original_score is None and reparsed is not None:
            recovered += 1

        records.append({
            "custom_id": f"row_{r.get('row_idx', i):06d}",
            "raw": raw,
            "score": score,
            "demographic": r.get("demographic", []),
            "referent_occ": r.get("referent_occ") or r.get("occupation", ""),
        })

    if recovered > 0:
        print(f"    Recovered {recovered} scores from raw text in {Path(results_file).name}")
    return records


def _parse_score(text):
    """Extract 0-10 score from text. Handles various formats from LLM output."""
    if not text:
        return None
    t = text.strip()

    # 1. Direct number: "7", "7.", "7/10"
    m = re.match(r"^(\d{1,2})(?:\.\d+)?(?:/10)?[.\s]*$", t.split("\n")[0].strip())
    if m:
        val = int(m.group(1))
        if 0 <= val <= 10:
            return val

    # 2. Number anywhere in first line: "Score: 7" or "Rating: 8/10"
    m = re.search(r"(?:score|rating|severity)[:\s]*(\d{1,2})", t.lower())
    if m:
        val = int(m.group(1))
        if 0 <= val <= 10:
            return val

    # 3. N/10 format anywhere: "4/10"
    m = re.search(r"\b(\d{1,2})\s*/\s*10\b", t)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 10:
            return val

    # 4. First standalone number in text
    m = re.search(r"\b(\d{1,2})\b", t)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 10:
            return val

    return None


# ═══════════════════════════════════════
# LOAD ORIGINAL CSV FOR IDENTITY MAPPING
# ═══════════════════════════════════════

def load_identity_mapping(csv_path):
    """Load original CSV to map row index → identity info."""
    mapping = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            demo_str = row.get("demographic_identifier", "[]")
            demo = [s.strip().strip("'\"") for s in demo_str.strip("[]").split(",") if s.strip()]
            mapping.append({
                "demographic": demo,
                "referent_occ": row.get("referent_occ", ""),
                "stereotype_label": row.get("stereotype_label", ""),
            })
    return mapping


# ═══════════════════════════════════════
# CONSOLIDATE ALL RESULTS
# ═══════════════════════════════════════

def _strip_chunk(name):
    """Remove _chunkNNN suffix to get base model name."""
    return re.sub(r"_chunk\d+$", "", name)


def collect_all_results(dataset_name):
    """
    Find and parse all result files for a dataset, merging chunks per model.

    Searches in:
      1. outputs/one_batch/{openai,anthropic}/   — run_one_batch results (JobFair, LBOX, Mind)
      2. outputs/local/{dataset}/                — run_one_experiment results
      3. outputs/batch/{openai,anthropic}/        — run_batch results (legacy, WinoIdentity)
      4. outputs/behavioral/{coref,job,legal}/    — run_experiment results (legacy)
    """
    all_model_results = {}

    # ── Active pipeline outputs ──

    # 1. OpenAI batch results (run_one_batch)
    openai_dir = OUTPUT_DIR / "one_batch" / "openai"
    if openai_dir.exists():
        for f in sorted(openai_dir.glob(f"results_{dataset_name}_*.jsonl")):
            raw_name = f.stem.replace(f"results_{dataset_name}_", "")
            model_name = _strip_chunk(raw_name)
            if model_name not in all_model_results:
                all_model_results[model_name] = []
            all_model_results[model_name].extend(parse_openai_results(f))

    # 2. Anthropic batch results (run_one_batch)
    anthropic_dir = OUTPUT_DIR / "one_batch" / "anthropic"
    if anthropic_dir.exists():
        for f in sorted(anthropic_dir.glob(f"results_{dataset_name}_*.jsonl")):
            raw_name = f.stem.replace(f"results_{dataset_name}_", "")
            model_name = _strip_chunk(raw_name)
            if model_name not in all_model_results:
                all_model_results[model_name] = []
            all_model_results[model_name].extend(parse_anthropic_results(f))

    # 3. Local model results (run_one_experiment)
    local_dir = OUTPUT_DIR / "local" / dataset_name
    if local_dir.exists():
        for f in sorted(local_dir.glob(f"raw_{dataset_name}_*.json")):
            model_name = f.stem.replace(f"raw_{dataset_name}_", "")
            all_model_results[model_name] = parse_local_results(f)

    # ── Legacy pipeline outputs (run_batch, run_experiment) ──

    # Map dataset names to legacy context prefixes
    legacy_prefix_map = {
        "winoidentity": "coref",
        "jobfair": "job",
        "lbox": "legal",
        "mind": "medical",
    }
    legacy_prefix = legacy_prefix_map.get(dataset_name, dataset_name)

    # 4. Legacy batch results: outputs/batch/{openai,anthropic}/results_{prefix}_{model}_chunk*.jsonl
    for provider in ["openai", "anthropic"]:
        legacy_batch = OUTPUT_DIR / "one_batch" / provider
        if legacy_batch.exists():
            for f in sorted(legacy_batch.glob(f"results_{legacy_prefix}_*.jsonl")):
                raw_name = f.stem.replace(f"results_{legacy_prefix}_", "")
                model_name = _strip_chunk(raw_name)
                if not model_name:
                    continue
                if model_name not in all_model_results:
                    all_model_results[model_name] = []
                parser = parse_openai_results if provider == "openai" else parse_anthropic_results
                all_model_results[model_name].extend(parser(f))

    # 5. Legacy experiment results: outputs/behavioral/{coref,job,legal}/raw_{prefix}_{model}.json
    legacy_dir = OUTPUT_DIR / "local"
    if legacy_dir.exists():
        for f in sorted(legacy_dir.glob(f"raw_{legacy_prefix}_*.json")):
            model_name = f.stem.replace(f"raw_{legacy_prefix}_", "")
            if model_name and model_name not in all_model_results:
                all_model_results[model_name] = parse_local_results(f)

    # 6. Pre-computed metrics (WinoIdentity has accuracy already computed)
    if legacy_dir and legacy_dir.exists():
        for f in sorted(legacy_dir.glob(f"metrics_{legacy_prefix}_*.json")):
            model_name = f.stem.replace(f"metrics_{legacy_prefix}_", "")
            if model_name:
                all_model_results[f"_metrics_{model_name}"] = f  # store path for later

    return all_model_results


# ═══════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════

def analyze_dataset(dataset_name, csv_path=None):
    """Full analysis pipeline for one dataset."""
    print(f"\n{'='*60}")
    print(f"  Analyzing: {dataset_name}")
    print(f"{'='*60}")

    out_dir = ANALYSIS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect results
    all_results = collect_all_results(dataset_name)
    if not all_results:
        print("  No results found.")
        return

    # Separate pre-computed metrics files from raw results
    precomputed_metrics = {}
    raw_results = {}
    for k, v in all_results.items():
        if k.startswith("_metrics_") and isinstance(v, Path):
            model_name = k.replace("_metrics_", "")
            precomputed_metrics[model_name] = v
        else:
            raw_results[k] = v

    is_coref = (dataset_name == "winoidentity")

    print(f"  Models found: {list(raw_results.keys())}")
    if precomputed_metrics:
        print(f"  Pre-computed metrics: {list(precomputed_metrics.keys())}")

    for mname, records in raw_results.items():
        if isinstance(records, list):
            if is_coref:
                # Count records with 'correct' field (coref) vs 'score' (scoring)
                has_correct = sum(1 for r in records if r.get("correct") is not None)
                has_score = sum(1 for r in records if r.get("score") is not None)
                print(f"    {mname}: {len(records)} records, {has_correct} with correct, {has_score} with score")
            else:
                scored = sum(1 for r in records if r.get("score") is not None)
                print(f"    {mname}: {len(records)} records, {scored} scored")

    # Load identity mapping
    identity_map = None
    if csv_path and Path(csv_path).exists():
        identity_map = load_identity_mapping(csv_path)
        print(f"  Identity mapping: {len(identity_map)} rows")

    # Per-model analysis
    summary = {}

    # 1. Load pre-computed metrics FIRST (WinoIdentity — already has accuracy)
    for mname, metrics_path in precomputed_metrics.items():
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            model_summary = _import_precomputed_metrics(mname, metrics)
            if model_summary:
                summary[mname] = model_summary
                print(f"    {mname}: loaded pre-computed metrics")
        except Exception as e:
            print(f"    {mname}: failed to load metrics: {e}")

    # 2. Analyze raw results (skip models already in summary from precomputed)
    for mname, records in raw_results.items():
        if mname in summary:
            continue  # precomputed metrics already loaded
        if not isinstance(records, list):
            continue
        # Attach identity info
        if identity_map:
            for r in records:
                idx_m = re.search(r"(\d+)", r.get("custom_id", ""))
                if idx_m:
                    idx = int(idx_m.group(1))
                    if idx < len(identity_map):
                        r["demographic"] = identity_map[idx]["demographic"]
                        r["referent_occ"] = identity_map[idx]["referent_occ"]
                        r["stereotype_label"] = identity_map[idx]["stereotype_label"]

        if is_coref:
            model_summary = _analyze_coref_model(mname, records)
        else:
            model_summary = _analyze_model(mname, records)
        if model_summary:
            summary[mname] = model_summary

    # Save summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print key findings
    if is_coref:
        _print_coref_findings(summary, dataset_name)
    else:
        _print_findings(summary, dataset_name)

    # Save consolidated CSV (only for scoring tasks)
    if not is_coref:
        _save_consolidated_csv(raw_results, identity_map, out_dir, dataset_name)

    return summary


def _analyze_model(mname, records):
    """Compute per-identity statistics for one model."""
    # Count total records per identity (including None scores)
    total_by_demo = defaultdict(int)
    scored_by_demo = defaultdict(list)
    for r in records:
        demo = r.get("demographic", [])
        key = "+".join(sorted(demo)) if demo else "(baseline)"
        total_by_demo[key] += 1
        if r.get("score") is not None:
            scored_by_demo[key].append(r["score"])

    per_identity = {}
    for key in sorted(set(list(total_by_demo.keys()) + list(scored_by_demo.keys()))):
        total = total_by_demo[key]
        scores = scored_by_demo.get(key, [])
        parts = key.split("+") if key != "(baseline)" else []
        missing = total - len(scores)
        missing_rate = missing / total if total > 0 else 0

        entry = {
            "mean_score": round(float(np.mean(scores)), 3) if scores else None,
            "std_score": round(float(np.std(scores)), 3) if scores else None,
            "median": float(np.median(scores)) if scores else None,
            "n": len(scores),
            "n_total": total,
            "n_missing": missing,
            "missing_rate": round(missing_rate, 3),
            "identity_type": "baseline" if not parts else ("single" if len(parts) == 1 else "multi"),
        }
        per_identity[key] = entry

    # Overall
    all_scores = [r["score"] for r in records if r.get("score") is not None]
    baseline_scores = scored_by_demo.get("(baseline)", [])
    total = len(records)
    scored = len(all_scores)
    missing = total - scored
    missing_rate = missing / total if total > 0 else 0

    overall = {
        "total_records": total,
        "scored": scored,
        "missing": missing,
        "missing_rate": round(missing_rate, 3),
        "mean_score": round(float(np.mean(all_scores)), 3) if all_scores else None,
        "baseline_mean": round(float(np.mean(baseline_scores)), 3) if baseline_scores else None,
    }

    # Flag identities with high missing rate (>10%)
    high_missing = {k: v for k, v in per_identity.items() if v["missing_rate"] > 0.10}
    if high_missing:
        overall["high_missing_identities"] = len(high_missing)
        overall["worst_missing"] = max(high_missing.items(), key=lambda x: x[1]["missing_rate"])

    # Disparity: difference between highest and lowest scored identity
    means = {k: v["mean_score"] for k, v in per_identity.items() if v["n"] >= 5 and v["mean_score"] is not None}
    if len(means) >= 2:
        overall["max_identity"] = max(means, key=means.get)
        overall["max_score"] = means[overall["max_identity"]]
        overall["min_identity"] = min(means, key=means.get)
        overall["min_score"] = means[overall["min_identity"]]
        overall["disparity"] = round(overall["max_score"] - overall["min_score"], 3)

    # Interaction terms for multi-identity
    i_terms = []
    baseline_mean = overall.get("baseline_mean")
    for key, stats in per_identity.items():
        if stats["identity_type"] != "multi":
            continue
        parts = key.split("+")
        if len(parts) != 2:
            continue
        s_a = per_identity.get(parts[0], {}).get("mean_score")
        s_b = per_identity.get(parts[1], {}).get("mean_score")
        s_multi = stats["mean_score"]
        if all(v is not None for v in [s_multi, s_a, s_b, baseline_mean]):
            I = round(s_multi - s_a - s_b + baseline_mean, 4)
            stats["I_term"] = I
            i_terms.append(I)

    if i_terms:
        overall["mean_I"] = round(float(np.mean(i_terms)), 4)
        overall["n_negative_I"] = sum(1 for v in i_terms if v < 0)
        overall["n_positive_I"] = sum(1 for v in i_terms if v > 0)
        overall["n_I"] = len(i_terms)

    return {"overall": overall, "per_identity": per_identity}


def _analyze_coref_model(mname, records):
    """Analyze WinoIdentity coref results: accuracy-based, not score-based."""
    by_demo = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in records:
        demo = r.get("demographic", [])
        key = "+".join(sorted(demo)) if demo else "(baseline)"

        # Parse choice from raw answer if 'correct' not already set
        correct = r.get("correct")
        if correct is None:
            raw = r.get("raw", "") or ""
            # Coref answers are "A 95" or "B 90" — extract choice
            choice = None
            for ch in raw.strip().upper()[:5]:
                if ch in "AB":
                    choice = ch
                    break
            if choice is not None:
                # We don't know ground truth from batch results alone
                # Mark as "responded" but can't compute accuracy
                by_demo[key]["total"] += 1
                r["_responded"] = True
                continue

        if correct is not None:
            by_demo[key]["total"] += 1
            if correct:
                by_demo[key]["correct"] += 1

    per_identity = {}
    for key, stats in sorted(by_demo.items()):
        total = stats["total"]
        correct = stats["correct"]
        acc = correct / total if total > 0 else None
        parts = key.split("+") if key != "(baseline)" else []
        per_identity[key] = {
            "accuracy": round(acc, 4) if acc is not None else None,
            "n_correct": correct,
            "n_total": total,
            "identity_type": "baseline" if not parts else ("single" if len(parts) == 1 else "multi"),
        }

    # Overall
    total_correct = sum(s["correct"] for s in by_demo.values())
    total_n = sum(s["total"] for s in by_demo.values())
    responded = sum(1 for r in records if r.get("correct") is not None or r.get("_responded"))

    overall = {
        "total_records": len(records),
        "responded": responded,
        "missing": len(records) - responded,
        "missing_rate": round((len(records) - responded) / len(records), 3) if records else 0,
        "overall_accuracy": round(total_correct / total_n, 4) if total_n > 0 else None,
        "n_correct": total_correct,
        "n_total": total_n,
        "task_type": "coref",
    }

    # Accuracy disparity
    accs = {k: v["accuracy"] for k, v in per_identity.items()
            if v["accuracy"] is not None and v["n_total"] >= 10}
    if len(accs) >= 2:
        overall["max_identity"] = max(accs, key=accs.get)
        overall["max_accuracy"] = accs[overall["max_identity"]]
        overall["min_identity"] = min(accs, key=accs.get)
        overall["min_accuracy"] = accs[overall["min_identity"]]
        overall["accuracy_gap"] = round(overall["max_accuracy"] - overall["min_accuracy"], 4)

    return {"overall": overall, "per_identity": per_identity}


def _import_precomputed_metrics(mname, metrics):
    """Import pre-computed metrics from run_experiment's metrics JSON."""
    # metrics format: {"(baseline)": {"accuracy": 0.87, "n": 3168}, "old": {"accuracy": 0.87, ...}, ...}
    per_identity = {}
    total_n = 0
    total_correct = 0
    for key, stats in metrics.items():
        if not isinstance(stats, dict) or "accuracy" not in stats:
            continue
        acc = stats["accuracy"]
        n = stats.get("n", 0)
        total_n += n
        total_correct += int(acc * n) if n > 0 else 0
        per_identity[key] = {
            "accuracy": round(acc, 4),
            "n_correct": int(acc * n),
            "n_total": n,
            "identity_type": stats.get("identity_type", "unknown"),
        }

    if not per_identity:
        return None

    overall = {
        "total_records": total_n,
        "responded": total_n,
        "missing": 0,
        "missing_rate": 0,
        "overall_accuracy": round(total_correct / total_n, 4) if total_n > 0 else None,
        "n_correct": total_correct,
        "n_total": total_n,
        "task_type": "coref",
        "source": "precomputed_metrics",
    }

    accs = {k: v["accuracy"] for k, v in per_identity.items()
            if v["accuracy"] is not None and v["n_total"] >= 10}
    if len(accs) >= 2:
        overall["max_identity"] = max(accs, key=accs.get)
        overall["max_accuracy"] = accs[overall["max_identity"]]
        overall["min_identity"] = min(accs, key=accs.get)
        overall["min_accuracy"] = accs[overall["min_identity"]]
        overall["accuracy_gap"] = round(overall["max_accuracy"] - overall["min_accuracy"], 4)

    return {"overall": overall, "per_identity": per_identity}


def _print_coref_findings(summary, dataset_name):
    """Print WinoIdentity coref findings: accuracy-based."""
    print(f"\n  --- Key Findings: {dataset_name} (coreference accuracy) ---")

    for mname, data in sorted(summary.items()):
        ov = data["overall"]
        print(f"\n  [{mname}]")
        mr = ov.get('missing_rate', 0)
        src = f" ({ov['source']})" if ov.get("source") else ""
        print(f"    Records: {ov.get('responded', '?')}/{ov['total_records']}{src}")
        print(f"    Overall accuracy: {ov.get('overall_accuracy')}")
        if ov.get("accuracy_gap"):
            print(f"    Accuracy gap: {ov['accuracy_gap']}")
            print(f"      Highest: {ov['max_identity']} ({ov['max_accuracy']:.4f})")
            print(f"      Lowest:  {ov['min_identity']} ({ov['min_accuracy']:.4f})")

        pi = data["per_identity"]
        ranked = {k: v for k, v in pi.items() if v.get("accuracy") is not None and v["n_total"] >= 10}
        if ranked:
            top = sorted(ranked.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:5]
            bottom = sorted(ranked.items(), key=lambda x: x[1]["accuracy"])[:5]
            print(f"    Top 5 highest accuracy:")
            for k, v in top:
                print(f"      {k:30s} {v['accuracy']:.4f} (n={v['n_total']})")
            print(f"    Top 5 lowest accuracy:")
            for k, v in bottom:
                print(f"      {k:30s} {v['accuracy']:.4f} (n={v['n_total']})")


def _print_findings(summary, dataset_name):
    """Print key findings across all models."""
    print(f"\n  --- Key Findings: {dataset_name} ---")

    for mname, data in summary.items():
        ov = data["overall"]
        print(f"\n  [{mname}]")
        mr = ov.get('missing_rate', 0)
        mr_pct = f"{mr*100:.1f}%"
        mr_flag = " ⚠️" if mr > 0.05 else ""
        print(f"    Records: {ov['scored']}/{ov['total_records']} scored (missing {ov.get('missing',0)}, {mr_pct}){mr_flag}")
        print(f"    Overall mean: {ov.get('mean_score')}  Baseline: {ov.get('baseline_mean')}")
        if ov.get("disparity"):
            print(f"    Disparity: {ov['disparity']}")
            print(f"      Highest: {ov['max_identity']} ({ov['max_score']})")
            print(f"      Lowest:  {ov['min_identity']} ({ov['min_score']})")
        if ov.get("mean_I") is not None:
            direction = "penalty (lower scores)" if ov["mean_I"] < 0 else "advantage (higher scores)"
            print(f"    Interaction: {ov['mean_I']} ({ov['n_negative_I']}/{ov['n_I']} negative = {direction})")

        # Missing rate by identity
        pi = data["per_identity"]
        high_miss = {k: v for k, v in pi.items() if v.get("missing_rate", 0) > 0.05}
        if high_miss:
            worst = sorted(high_miss.items(), key=lambda x: x[1]["missing_rate"], reverse=True)
            print(f"    ⚠️  High missing rate identities ({len(worst)}):")
            for k, v in worst[:10]:
                print(f"      {k:30s} {v['n']}/{v['n_total']} scored ({v['missing_rate']*100:.0f}% missing)")

        # Top 5 most/least favored
        scored = {k: v for k, v in pi.items() if v["n"] >= 5 and k != "(baseline)" and v.get("mean_score") is not None}
        if scored:
            top = sorted(scored.items(), key=lambda x: x[1]["mean_score"], reverse=True)[:5]
            bottom = sorted(scored.items(), key=lambda x: x[1]["mean_score"])[:5]
            print(f"    Top 5 highest scored:")
            for k, v in top:
                print(f"      {k:30s} {v['mean_score']:.2f} ± {v['std_score']:.2f} (n={v['n']}/{v['n_total']})")
            print(f"    Top 5 lowest scored:")
            for k, v in bottom:
                print(f"      {k:30s} {v['mean_score']:.2f} ± {v['std_score']:.2f} (n={v['n']}/{v['n_total']})")


def _save_consolidated_csv(all_results, identity_map, out_dir, dataset_name):
    """Save a single CSV with all models' scores aligned by row."""
    models = sorted(all_results.keys())
    if not models:
        return

    # Determine max rows
    max_rows = max(len(records) for records in all_results.values())

    rows = []
    for i in range(max_rows):
        row = {"row_idx": i}
        if identity_map and i < len(identity_map):
            row["demographic"] = "+".join(identity_map[i]["demographic"]) or "(baseline)"
            row["referent_occ"] = identity_map[i]["referent_occ"]
            row["stereotype_label"] = identity_map[i]["stereotype_label"]

        for mname in models:
            records = all_results[mname]
            if i < len(records):
                row[f"score_{mname}"] = records[i].get("score")
                row[f"raw_{mname}"] = records[i].get("raw", "")
            else:
                row[f"score_{mname}"] = None
                row[f"raw_{mname}"] = ""
        rows.append(row)

    fieldnames = ["row_idx", "demographic", "referent_occ", "stereotype_label"]
    for m in models:
        fieldnames.extend([f"score_{m}", f"raw_{m}"])

    csv_file = out_dir / f"consolidated_{dataset_name}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {csv_file}")


# ═══════════════════════════════════════
# CLI
# ═══════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Analyze experiment results")
    p.add_argument("--dataset", choices=["jobfair", "lbox", "mind", "winoidentity"])
    p.add_argument("--csv", type=str, default=None, help="Original dataset CSV for identity mapping")
    p.add_argument("--all", action="store_true", help="Analyze all datasets")
    args = p.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = ["jobfair", "lbox", "mind", "winoidentity"] if args.all else [args.dataset]
    if not args.all and not args.dataset:
        print("Error: --dataset or --all required")
        sys.exit(1)

    default_csvs = {
        "jobfair": "data/jobfair.csv",
        "lbox": "data/lbox.csv",
        "mind": "data/mind.csv",
        "winoidentity": "data/winoidentity.csv",
    }

    for ds in datasets:
        if ds is None:
            continue
        csv_path = args.csv or default_csvs.get(ds)
        analyze_dataset(ds, csv_path)


if __name__ == "__main__":
    main()