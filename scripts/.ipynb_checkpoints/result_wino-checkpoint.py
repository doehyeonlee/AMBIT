"""
WinoIdentity (Coreference) result extraction and analysis.

Collects results from three sources:
  1. Precomputed metrics: outputs/behavioral/coref/metrics_coref_{model}.json
  2. Precomputed raw:     outputs/behavioral/coref/raw_coref_{model}.json
  3. API batch results:   outputs/batch/{openai,anthropic}/results_coref_*.jsonl

Parses batch results → accuracy per identity, then combines all sources
into a unified analysis comparable with JobFair/LBOX.

Usage:
  python -m scripts.result_wino
  python -m scripts.result_wino --output-dir outputs/figures
"""

import json, csv, argparse, sys, re, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats as sp

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"

GENDERS = ["male", "female"]
TRAIT_CATEGORIES = {
    "age": ["young", "old"], "disability": ["able-bodied", "disabled"],
    "body": ["thin", "fat"], "neurodiv": ["neurotypical", "neurodivergent"],
    "nationality": ["american", "immigrant"], "language": ["english-speaking", "non-english-speaking"],
    "sexuality": ["heterosexual", "gay"], "gender_id": ["cisgender", "transgender"],
    "ses": ["rich", "poor"], "race": ["white", "black", "asian", "hispanic"],
    "religion": ["christian", "muslim", "jewish"],
}
ALL_TRAITS = [t for ts in TRAIT_CATEGORIES.values() for t in ts]


# ═══════════════════════════════════════
# 1. LOAD WINOIDENTITY CSV (ground truth)
# ═══════════════════════════════════════

def load_winoidentity_csv(filepath):
    """Load WinoIdentity CSV with ground truth for coreference.
    Extracts gender from pronoun column (she/her→female, he/him/his→male)
    and combines with demographic_identifier to create gender+trait keys.
    """
    FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}
    MALE_PRONOUNS = {"he", "him", "his", "himself"}

    probes = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            occs = row.get("occs_in_samples", "[]")
            if isinstance(occs, str):
                occs = [s.strip().strip("'\"") for s in occs.strip("[]").split(",") if s.strip()]
            demo = row.get("demographic_identifier", "[]")
            if isinstance(demo, str):
                demo = [s.strip().strip("'\"") for s in demo.strip("[]").split(",") if s.strip()]

            # Extract gender from pronoun
            pronoun = row.get("pronoun", "").strip().lower()
            if pronoun in FEMALE_PRONOUNS:
                gender = "female"
            elif pronoun in MALE_PRONOUNS:
                gender = "male"
            else:
                gender = None

            # Build combined demographic: gender + trait
            # e.g. pronoun="she", demo=["fat"] → combined=["female", "fat"]
            combined = []
            if gender:
                combined.append(gender)
            combined.extend(demo)

            probes.append({
                "referent_occ": row.get("referent_occ", "").strip(),
                "occs_in_samples": occs,
                "demographic": combined,  # now includes gender from pronoun
                "demographic_trait_only": demo,  # original trait without gender
                "gender": gender,
                "stereotype_label": row.get("stereotype_label", ""),
                "task_type": row.get("winobias_task_type", ""),
            })
    return probes


def get_correct_answer(probe):
    """Given a probe, return the correct choice label ('A' or 'B')."""
    occs = probe["occs_in_samples"]
    ref = probe["referent_occ"]
    if ref in occs:
        return chr(65 + occs.index(ref))  # A=0, B=1
    return None


# ═══════════════════════════════════════
# 2. PARSE BATCH RESULTS
# ═══════════════════════════════════════

def parse_choice(raw):
    """Parse 'A 85' or 'B' → (choice, confidence)."""
    if not raw:
        return None, None
    raw = raw.strip()
    m = re.match(r"([A-Ba-b])\s*[:\-\(\[]?\s*(\d{1,3})", raw)
    if m:
        return m.group(1).upper(), int(m.group(2))
    for ch in raw.upper()[:5]:
        if ch in "AB":
            return ch, None
    return None, None


def parse_openai_batch(filepath):
    """Parse OpenAI batch results → list of (custom_id, raw_text)."""
    records = []
    errors = 0
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            cid = data.get("custom_id", "")
            body = data.get("response", {}).get("body", {})
            choices = body.get("choices", [])
            text = choices[0]["message"]["content"].strip() if choices else None
            records.append({"custom_id": cid, "raw": text})
    if errors:
        print(f"    ⚠️ {errors} corrupt lines skipped in {filepath.name}")
    return records


def parse_anthropic_batch(filepath):
    """Parse Anthropic batch results → list of (custom_id, raw_text)."""
    records = []
    errors = 0
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            data = json.loads(line)
            cid = data.get("custom_id", "")
            result = data.get("result", {})
            text = None
            if result.get("type") == "succeeded":
                content = result.get("message", {}).get("content", [])
                text = content[0]["text"].strip() if content else None
            records.append({"custom_id": cid, "raw": text})
    return records


def process_batch_results(batch_records, probes):
    """
    Match batch records to probes using custom_id index,
    compute correct/incorrect per record.
    Returns list of {probe_idx, demographic, chosen, correct, confidence}.
    """
    results = []
    for rec in batch_records:
        # Extract probe index from custom_id: "coref_000123" → 123
        m = re.search(r"(\d+)", rec["custom_id"])
        if not m:
            continue
        idx = int(m.group(1))
        if idx >= len(probes):
            continue

        probe = probes[idx]
        chosen, confidence = parse_choice(rec["raw"])
        correct_label = get_correct_answer(probe)

        correct = None
        if chosen and correct_label:
            correct = (chosen == correct_label)

        results.append({
            "probe_idx": idx,
            "demographic": probe["demographic"],
            "referent_occ": probe["referent_occ"],
            "stereotype_label": probe["stereotype_label"],
            "task_type": probe["task_type"],
            "chosen": chosen,
            "correct": correct,
            "confidence": confidence,
            "raw": rec["raw"],
        })
    return results


# ═══════════════════════════════════════
# 3. COMPUTE METRICS
# ═══════════════════════════════════════

def compute_metrics(results):
    """Compute per-identity accuracy + interaction terms from raw results.
    For WinoIdentity where every probe has gender+trait, also computes
    single-axis accuracy by aggregating across the other axis.
    """
    by_demo = defaultdict(list)
    for r in results:
        demo = r.get("demographic", [])
        key = "+".join(sorted(demo)) if demo else "(baseline)"
        by_demo[key].append(r)

    per_identity = {}
    for key, group in sorted(by_demo.items()):
        valid = [r for r in group if r.get("correct") is not None]
        if not valid:
            per_identity[key] = {"accuracy": None, "n": len(group), "n_valid": 0}
            continue
        acc = sum(1 for r in valid if r["correct"]) / len(valid)
        parts = key.split("+") if key != "(baseline)" else []
        per_identity[key] = {
            "accuracy": round(acc, 4),
            "n": len(group),
            "n_valid": len(valid),
            "n_correct": sum(1 for r in valid if r["correct"]),
            "identity_type": "baseline" if not parts else ("single" if len(parts) == 1 else "multi"),
        }

    # Compute single-axis accuracy by aggregating across the other axis
    # e.g., acc("fat") = mean accuracy across all probes where "fat" is in demographic
    by_single = defaultdict(list)
    for r in results:
        if r.get("correct") is None:
            continue
        demo = r.get("demographic", [])
        for part in demo:
            by_single[part].append(r)

    for trait, group in by_single.items():
        if trait in per_identity:
            continue  # already exists as a standalone key
        valid = [r for r in group if r.get("correct") is not None]
        if not valid:
            continue
        acc = sum(1 for r in valid if r["correct"]) / len(valid)
        per_identity[trait] = {
            "accuracy": round(acc, 4),
            "n": len(group),
            "n_valid": len(valid),
            "n_correct": sum(1 for r in valid if r["correct"]),
            "identity_type": "single",
            "aggregated": True,  # marker: this was computed by aggregation
        }

    # Interaction terms: I = acc_multi - acc_a - acc_b + acc_base
    # Use overall mean as pseudo-baseline if no explicit baseline
    base_acc = per_identity.get("(baseline)", {}).get("accuracy")
    if base_acc is None:
        all_valid = [r for r in results if r.get("correct") is not None]
        base_acc = sum(1 for r in all_valid if r["correct"]) / len(all_valid) if all_valid else None
        if base_acc is not None:
            per_identity["(baseline)"] = {
                "accuracy": round(base_acc, 4), "n": len(all_valid), "n_valid": len(all_valid),
                "identity_type": "baseline", "aggregated": True,
            }

    for key, m in per_identity.items():
        if m.get("identity_type") != "multi" or m.get("accuracy") is None:
            continue
        parts = key.split("+")
        if len(parts) != 2:
            continue
        acc_a = per_identity.get(parts[0], {}).get("accuracy")
        acc_b = per_identity.get(parts[1], {}).get("accuracy")
        if all(v is not None for v in [m["accuracy"], acc_a, acc_b, base_acc]):
            m["I_accuracy"] = round(m["accuracy"] - acc_a - acc_b + base_acc, 4)

    # Overall
    all_valid = [r for r in results if r.get("correct") is not None]
    overall_acc = sum(1 for r in all_valid if r["correct"]) / len(all_valid) if all_valid else None

    accs = {k: v["accuracy"] for k, v in per_identity.items() if v.get("accuracy") is not None and v["n_valid"] >= 10}
    i_vals = [v["I_accuracy"] for v in per_identity.values() if v.get("I_accuracy") is not None]

    overall = {
        "total": len(results),
        "valid": len(all_valid),
        "overall_accuracy": round(overall_acc, 4) if overall_acc else None,
        "n_identities": len(per_identity),
    }
    if len(accs) >= 2:
        overall["accuracy_disparity"] = round(max(accs.values()) - min(accs.values()), 4)
        overall["most_accurate"] = max(accs, key=accs.get)
        overall["least_accurate"] = min(accs, key=accs.get)
    if i_vals:
        t_stat, p_val = sp.ttest_1samp(i_vals, 0)
        overall["mean_I"] = round(float(np.mean(i_vals)), 4)
        overall["std_I"] = round(float(np.std(i_vals, ddof=1)), 4)
        overall["I_t_stat"] = round(float(t_stat), 4)
        overall["I_p_value"] = float(p_val)
        overall["n_negative_I"] = sum(1 for v in i_vals if v < 0)
        overall["n_total_I"] = len(i_vals)

    return {"per_identity": per_identity, "overall": overall}


# ═══════════════════════════════════════
# 4. COLLECT ALL SOURCES
# ═══════════════════════════════════════

def collect_all(probes):
    """Collect WinoIdentity results from all sources."""
    all_models = {}  # model_name → {"metrics": ..., "source": ...}

    # 1. Precomputed raw → recompute with gender from probes
    #    (Preferred over precomputed metrics because we now extract gender from pronoun)
    coref_dir = OUTPUT_DIR / "local"
    if coref_dir.exists() and probes:
        for f in sorted(coref_dir.glob("raw_coref_*.json")):
            model = f.stem.replace("raw_coref_", "")
            try:
                with open(f) as fh:
                    raw = json.load(fh)
                if isinstance(raw, list) and raw and isinstance(raw[0], dict):
                    # Inject gender from probes into raw results
                    for r in raw:
                        idx = r.get("probe_idx", -1)
                        if 0 <= idx < len(probes):
                            r["demographic"] = probes[idx]["demographic"]  # now includes gender
                    metrics = compute_metrics(raw)
                    all_models[model] = {"metrics": metrics, "source": "raw_recomputed_with_gender"}
                    print(f"  Recomputed with gender: {model}")
            except Exception as e:
                print(f"  Failed raw: {model} — {e}")

    # 2. Precomputed metrics (fallback — no gender info)
    if coref_dir.exists():
        for f in sorted(coref_dir.glob("metrics_coref_*.json")):
            model = f.stem.replace("metrics_coref_", "")
            if model in all_models:
                continue  # already recomputed from raw
            try:
                with open(f) as fh:
                    metrics = json.load(fh)
                all_models[model] = {"metrics": metrics, "source": "precomputed_no_gender"}
                print(f"  Loaded precomputed (no gender): {model}")
            except Exception as e:
                print(f"  Failed: {model} — {e}")

    # 3. Batch results (API models)
    for provider in ["openai", "anthropic"]:
        batch_dir = OUTPUT_DIR / "one_batch" / provider
        if not batch_dir.exists():
            continue

        # Group chunk files by model
        by_model = defaultdict(list)
        for f in sorted(batch_dir.glob("results_coref_*.jsonl")):
            stem = f.stem.replace("results_coref_", "")
            # Strip _chunkNNN
            model = re.sub(r"_chunk\d+$", "", stem)
            by_model[model].append(f)

        for model, files in sorted(by_model.items()):
            if model in all_models:
                continue  # precomputed takes priority
            print(f"  Parsing batch: {model} ({len(files)} chunks, {provider})...")

            all_records = []
            parser = parse_openai_batch if provider == "openai" else parse_anthropic_batch
            for f in files:
                all_records.extend(parser(f))

            if not all_records:
                continue

            # Match with probes and compute accuracy
            results = process_batch_results(all_records, probes)
            if not results:
                continue

            metrics = compute_metrics(results)
            valid = sum(1 for r in results if r.get("correct") is not None)
            answered = sum(1 for r in results if r.get("chosen"))
            print(f"    {len(all_records)} records → {answered} answered → {valid} validated")

            all_models[model] = {"metrics": metrics, "source": f"batch_{provider}"}

    # Filter out test-only models (< 100 records)
    filtered = {}
    for model, data in all_models.items():
        m = data["metrics"]
        ov = m.get("overall", m)
        total = ov.get("total", ov.get("valid", 0))
        if total < 100:
            print(f"  Skipping {model}: only {total} records (test run)")
            continue
        if ov.get("overall_accuracy") is None:
            print(f"  Skipping {model}: no accuracy computed")
            continue
        filtered[model] = data

    return filtered


# ═══════════════════════════════════════
# 5. CROSS-CONTEXT COMPARISON
# ═══════════════════════════════════════

def cross_context_with_scoring(wino_models, out_dir):
    """Compare WinoIdentity accuracy deviation with JobFair/LBOX score deviation."""
    plt = setup_plt()

    # Load JobFair and LBOX summaries
    scoring_data = {}
    for ds in ["jobfair", "lbox"]:
        path = ANALYSIS_DIR / ds / "summary.json"
        if path.exists():
            with open(path) as f:
                scoring_data[ds] = json.load(f)

    if not scoring_data:
        print("  No JobFair/LBOX summaries found for cross-context comparison")
        return {}

    results = {}

    for wmodel, wdata in wino_models.items():
        pi = wdata["metrics"]
        # Handle both precomputed (flat) and recomputed (nested) formats
        if "per_identity" in pi:
            pi = pi["per_identity"]

        base_acc = _get_baseline_acc(pi)
        if base_acc is None:
            continue

        # Trait-level accuracy deviation
        trait_acc_dev = {}
        for key, stats in pi.items():
            if not isinstance(stats, dict) or stats.get("accuracy") is None or key == "(baseline)":
                continue
            parts = key.split("+")
            for p in parts:
                if p in ALL_TRAITS:
                    if p not in trait_acc_dev:
                        trait_acc_dev[p] = []
                    trait_acc_dev[p].append(stats["accuracy"] - base_acc)
        trait_acc_dev = {t: np.mean(vs) for t, vs in trait_acc_dev.items() if len(vs) >= 2}

        # Compare with scoring datasets
        for ds, summary in scoring_data.items():
            # Find matching model in scoring summary
            score_model = None
            for sk in summary.keys():
                if wmodel in sk or sk in wmodel or wmodel.replace("-", "") == sk.replace("-", ""):
                    score_model = sk
                    break
            if not score_model:
                print(f"    {wmodel}: no matching model in {ds} summary (keys: {list(summary.keys())[:5]})")
                continue

            spi = summary[score_model].get("per_identity", {})
            s_bl = spi.get("(baseline)", {}).get("mean_score")
            if s_bl is None:
                continue

            # Trait-level score deviation
            trait_score_dev = {}
            for key, stats in spi.items():
                if stats.get("mean_score") is None or key == "(baseline)":
                    continue
                parts = key.split("+")
                for p in parts:
                    if p in ALL_TRAITS:
                        if p not in trait_score_dev:
                            trait_score_dev[p] = []
                        dev = stats["mean_score"] - s_bl
                        # LBOX: flip sign (higher score = worse for defendant)
                        if ds == "lbox":
                            dev = -dev
                        trait_score_dev[p].append(dev)
            trait_score_dev = {t: np.mean(vs) for t, vs in trait_score_dev.items() if len(vs) >= 2}

            # Common traits
            common = sorted(set(trait_acc_dev.keys()) & set(trait_score_dev.keys()))
            if len(common) < 5:
                print(f"    {wmodel} ↔ {ds}: only {len(common)} common traits "
                      f"(wino={len(trait_acc_dev)}, score={len(trait_score_dev)}). Skipping.")
                continue

            x = [trait_acc_dev[t] for t in common]
            y = [trait_score_dev[t] for t in common]
            rho, p_val = sp.spearmanr(x, y)

            ds_label = "JobFair (score)" if ds == "jobfair" else "LBOX (severity, flipped)"
            print(f"    {wmodel} WinoIdentity ↔ {ds_label} [{score_model}]: ρ={rho:.3f}, p={p_val:.4f}")

            key = f"{wmodel}_{ds}"
            results[key] = {
                "wino_model": wmodel, "score_model": score_model, "dataset": ds,
                "spearman_rho": round(float(rho), 4), "p_value": float(p_val),
                "n_traits": len(common),
                "traits": [{"trait": t, "acc_dev": round(trait_acc_dev[t], 4),
                           "score_dev": round(trait_score_dev[t], 4)} for t in common],
            }

            # Plot
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.scatter(x, y, s=40, alpha=0.7)
            for i, t in enumerate(common):
                ax.annotate(t, (x[i], y[i]), fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=0.5)
            ax.set_xlabel("WinoIdentity: Accuracy Deviation from Baseline")
            ax.set_ylabel(f"{ds_label}: Score Deviation (positive = favorable)")
            ax.set_title(f"3-Context: {wmodel}\nWinoIdentity ↔ {ds} (ρ={rho:.3f}, p={p_val:.4f})")
            plt.tight_layout()
            fig.savefig(out_dir / f"cross3_{wmodel}_{ds}.png")
            plt.close(fig)

    return results


# ═══════════════════════════════════════
# 6. PRINT & SAVE
# ═══════════════════════════════════════

def setup_plt():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
    return plt


def print_summary(all_models, out_dir):
    """Print and save WinoIdentity summary."""
    plt = setup_plt()

    print(f"\n{'='*60}")
    print("  WinoIdentity Coreference Results")
    print(f"{'='*60}")

    summary_all = {}

    for model, data in sorted(all_models.items()):
        m = data["metrics"]
        ov = m.get("overall", m)  # precomputed may be flat
        pi = m.get("per_identity", m)  # or nested
        src = data["source"]

        print(f"\n  [{model}] (source: {src})")
        print(f"    Overall accuracy: {ov.get('overall_accuracy', 'N/A')}")
        print(f"    Records: {ov.get('total', ov.get('valid', '?'))}")

        if ov.get("accuracy_disparity"):
            print(f"    Accuracy disparity: {ov['accuracy_disparity']:.4f}")
            print(f"      Best:  {ov.get('most_accurate', '?')}")
            print(f"      Worst: {ov.get('least_accurate', '?')}")

        if ov.get("mean_I") is not None:
            p_str = f"p={ov['I_p_value']:.4f}" if ov['I_p_value'] >= 0.0001 else "p<0.0001"
            print(f"    Interaction: mean(I)={ov['mean_I']:+.4f} "
                  f"({ov['n_negative_I']}/{ov['n_total_I']} negative) "
                  f"t={ov['I_t_stat']:.2f} {p_str}")

        # Top/bottom accuracy
        accs = {k: v["accuracy"] for k, v in pi.items()
                if isinstance(v, dict) and v.get("accuracy") is not None
                and v.get("n", v.get("n_valid", 0)) >= 10}
        if accs:
            top = sorted(accs.items(), key=lambda x: x[1], reverse=True)[:5]
            bottom = sorted(accs.items(), key=lambda x: x[1])[:5]
            print(f"    Top 5 accuracy:")
            for k, v in top:
                n = pi[k].get("n_valid", pi[k].get("n", 0))
                print(f"      {k:30s} {v:.4f} (n={n})")
            print(f"    Bottom 5 accuracy:")
            for k, v in bottom:
                n = pi[k].get("n_valid", pi[k].get("n", 0))
                print(f"      {k:30s} {v:.4f} (n={n})")

        summary_all[model] = {"overall": ov, "source": src}

        # Plot: accuracy by trait
        base_acc = _get_baseline_acc(pi)
        if base_acc is not None:
            trait_devs = {}
            for key, stats in pi.items():
                if not isinstance(stats, dict) or stats.get("accuracy") is None or key == "(baseline)":
                    continue
                parts = key.split("+")
                for p_name in parts:
                    if p_name in ALL_TRAITS:
                        if p_name not in trait_devs:
                            trait_devs[p_name] = []
                        trait_devs[p_name].append(stats["accuracy"] - base_acc)

            if trait_devs:
                trait_means = {t: np.mean(vs) for t, vs in trait_devs.items()}
                sorted_traits = sorted(trait_means.items(), key=lambda x: x[1])
                names = [t[0] for t in sorted_traits]
                vals = [t[1] for t in sorted_traits]

                fig, ax = plt.subplots(figsize=(12, 8))
                colors = ["#d32f2f" if v < -0.005 else "#388e3c" if v > 0.005 else "#757575" for v in vals]
                ax.barh(range(len(names)), vals, color=colors, alpha=0.8)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names, fontsize=9)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("Accuracy Deviation from Baseline")
                ax.set_title(f"WinoIdentity: {model} — Trait Accuracy Deviation")
                plt.tight_layout()
                fig.savefig(out_dir / f"wino_traits_{model}.png")
                plt.close(fig)
                print(f"    Saved: wino_traits_{model}.png")

    # Save
    save_path = out_dir / "winoidentity_summary.json"
    with open(save_path, "w") as f:
        json.dump(summary_all, f, indent=2, default=str)
    print(f"\n  Summary saved: {save_path}")

    return summary_all


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="WinoIdentity coreference result analysis")
    p.add_argument("--data-file", default=None, help="WinoIdentity CSV (default: data/winoidentity.csv)")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load WinoIdentity CSV (ground truth)
    data_file = args.data_file or str(DATA_DIR / "winoidentity.csv")
    if not Path(data_file).exists():
        print(f"WinoIdentity CSV not found: {data_file}")
        print("Proceeding with precomputed metrics only (no batch parsing).")
        probes = None
    else:
        probes = load_winoidentity_csv(data_file)
        print(f"Loaded {len(probes)} probes from {data_file}")

    # Collect all results
    print("\nCollecting WinoIdentity results...")
    all_models = collect_all(probes or [])
    if not all_models:
        print("No WinoIdentity results found.")
        sys.exit(1)

    # Print and save summary
    summary = print_summary(all_models, out_dir)

    # ── Behavioral-style analysis (parallel to result_ans.py) ──
    print(f"\n{'='*60}")
    print("  Behavioral-Style Analysis (Accuracy-Based)")
    print(f"{'='*60}")
    section_wino_baseline(all_models)
    section_wino_trait_deviation(all_models, out_dir)
    section_wino_interaction(all_models)
    section_wino_model_comparison(all_models, out_dir)

    # Cross-context comparison
    if any(ds_dir.exists() for ds_dir in [ANALYSIS_DIR / "jobfair", ANALYSIS_DIR / "lbox"]):
        print(f"\n{'='*60}")
        print("  Cross-Context: WinoIdentity ↔ JobFair/LBOX")
        print(f"{'='*60}")
        cross = cross_context_with_scoring(all_models, out_dir)
        if cross:
            with open(out_dir / "cross_3context.json", "w") as f:
                json.dump(cross, f, indent=2, default=str)
            print(f"\n  Cross-context saved: {out_dir / 'cross_3context.json'}")
    else:
        print("\n  No JobFair/LBOX summaries found — skipping cross-context comparison")

    # SAE cross-reference
    sae_dir = OUTPUT_DIR / "sae"
    wino_sae_dirs = sorted(sae_dir.glob("winoidentity_*")) if sae_dir.exists() else []
    if wino_sae_dirs:
        print(f"\n{'='*60}")
        print("  SAE ↔ WinoIdentity Accuracy")
        print(f"{'='*60}")
        section_wino_sae(all_models, wino_sae_dirs, out_dir)
    else:
        print(f"\n  No WinoIdentity SAE results found in {sae_dir}.")
        print(f"  Run: python -m scripts.run_sae_analysis --data-file data/winoidentity.csv --model gemma-2-9b --multi-layer")


# ═══════════════════════════════════════
# BEHAVIORAL-STYLE ANALYSIS SECTIONS
# ═══════════════════════════════════════

def _get_pi(data):
    """Extract per_identity dict from model data, handling different formats."""
    m = data["metrics"]
    # Format 1: compute_metrics output: {"per_identity": {...}, "overall": {...}}
    if "per_identity" in m:
        return m["per_identity"]
    # Format 2: precomputed metrics from run_experiment: {"(baseline)": {...}, "old": {...}, ...}
    #   or just {"old": {...}, "transgender": {...}, ...}
    if isinstance(m, dict) and any(isinstance(v, dict) and "accuracy" in v for v in m.values()):
        return m
    # Format 3: overall-only (from winoidentity_summary.json)
    if "overall" in m:
        return {}
    return m


def _get_baseline_acc(pi):
    """Get baseline accuracy. If no explicit baseline, use overall mean."""
    # Try explicit baseline
    base = pi.get("(baseline)", {})
    if base.get("accuracy") is not None:
        return base["accuracy"]
    # Compute mean across all identities as pseudo-baseline
    accs = [v["accuracy"] for v in pi.values()
            if isinstance(v, dict) and v.get("accuracy") is not None]
    return float(np.mean(accs)) if accs else None


def section_wino_baseline(all_models):
    """[A] Baseline validation: does accuracy differ by single identity vs baseline?"""
    print(f"\n  --- [A] Baseline Accuracy Validation ---")

    for model, data in sorted(all_models.items()):
        pi = _get_pi(data)
        if not pi:
            continue
        base_acc = _get_baseline_acc(pi)
        if base_acc is None:
            continue

        # Get total n for baseline (or estimate)
        base_entry = pi.get("(baseline)", {})
        base_n = base_entry.get("n", base_entry.get("n_valid", 0))
        if base_n < 10:
            # Estimate from overall
            ov = data["metrics"].get("overall", {})
            total = ov.get("total", ov.get("valid", 0))
            n_ids = ov.get("n_identities", len(pi))
            base_n = total // max(n_ids, 1) if total > 0 else 300

        sig_count = 0
        total_count = 0
        worst_trait = None
        worst_diff = 0

        for key, stats in sorted(pi.items()):
            if key == "(baseline)" or not isinstance(stats, dict):
                continue
            acc = stats.get("accuracy")
            n = stats.get("n_valid", stats.get("n", 0))
            if acc is None or n < 10:
                continue

            total_count += 1
            p_hat = (base_acc * base_n + acc * n) / (base_n + n)
            if 0 < p_hat < 1:
                se = np.sqrt(p_hat * (1 - p_hat) * (1/base_n + 1/n))
                z = (acc - base_acc) / se if se > 0 else 0
                p_val = 2 * (1 - sp.norm.cdf(abs(z)))
                if p_val < 0.05:
                    sig_count += 1
                diff = acc - base_acc
                if abs(diff) > abs(worst_diff):
                    worst_diff = diff
                    worst_trait = key

        has_baseline = "(baseline)" in pi and pi["(baseline)"].get("accuracy") is not None
        bl_label = f"baseline_acc={base_acc:.4f}" if has_baseline else f"mean_acc={base_acc:.4f}(pseudo)"
        print(f"    {model:20s} {bl_label}(n≈{base_n})  "
              f"{sig_count}/{total_count} traits significantly differ  "
              f"worst={worst_trait}({worst_diff:+.4f})" if worst_trait else
              f"    {model:20s} {bl_label} — no traits to compare")


def section_wino_trait_deviation(all_models, out_dir):
    """[E-analog] Trait-level accuracy deviation from baseline, all models."""
    print(f"\n  --- [E] Trait Accuracy Deviation (all models) ---")
    plt = setup_plt()

    model_data = {}
    for model, data in sorted(all_models.items()):
        pi = _get_pi(data)
        base_acc = _get_baseline_acc(pi)
        if base_acc is None:
            continue

        devs = {}
        for key, stats in pi.items():
            if key == "(baseline)" or not isinstance(stats, dict):
                continue
            acc = stats.get("accuracy")
            if acc is None:
                continue
            # Key might be single trait or multi. Extract traits.
            parts = key.split("+")
            for p_name in parts:
                if p_name in ALL_TRAITS or p_name in GENDERS:
                    if p_name not in devs:
                        devs[p_name] = []
                    devs[p_name].append(acc - base_acc)

        trait_means = {t: np.mean(vs) for t, vs in devs.items() if len(vs) >= 2}
        if not trait_means:
            continue

        model_data[model] = trait_means

        sorted_t = sorted(trait_means.items(), key=lambda x: x[1])
        print(f"\n    [{model}] baseline={base_acc:.4f}")
        print(f"      Bottom 3: {', '.join(f'{t}({v:+.4f})' for t,v in sorted_t[:3])}")
        print(f"      Top 3:    {', '.join(f'{t}({v:+.4f})' for t,v in sorted_t[-3:])}")

    # Cross-model heatmap
    if len(model_data) >= 2:
        models = sorted(model_data.keys())
        all_trait_names = sorted(set(t for m in model_data.values() for t in m.keys()))
        matrix = np.zeros((len(all_trait_names), len(models)))
        for mi, model in enumerate(models):
            for ti, trait in enumerate(all_trait_names):
                matrix[ti, mi] = model_data[model].get(trait, 0)

        fig, ax = plt.subplots(figsize=(14, 10))
        vmax = max(abs(matrix.min()), abs(matrix.max())) or 0.05
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(all_trait_names)))
        ax.set_yticklabels(all_trait_names, fontsize=8)
        ax.set_title("WinoIdentity: Trait Accuracy Deviation from Baseline (all models)")
        plt.colorbar(im, ax=ax, shrink=0.6, label="Accuracy Deviation")
        plt.tight_layout()
        path = out_dir / "wino_heatmap_all_models.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"\n    Saved: {path}")


def section_wino_interaction(all_models):
    """[C-analog] Test interaction terms: I = acc_multi - acc_a - acc_b + acc_base."""
    print(f"\n  --- [C] Interaction Term Analysis ---")

    for model, data in sorted(all_models.items()):
        pi = _get_pi(data)
        base_acc = _get_baseline_acc(pi)
        if base_acc is None:
            continue

        i_vals = []
        for key, stats in pi.items():
            it = stats.get("identity_type") if isinstance(stats, dict) else None
            i_acc = stats.get("I_accuracy") if isinstance(stats, dict) else None
            if i_acc is not None:
                i_vals.append({"identity": key, "I": i_acc})

        if len(i_vals) < 5:
            # WinoIdentity may only have single traits (no multi)
            print(f"    {model:20s} — insufficient multi-identity data for interaction analysis")
            continue

        vals = [x["I"] for x in i_vals]
        t_stat, p_val = sp.ttest_1samp(vals, 0)
        n_neg = sum(1 for v in vals if v < 0)

        print(f"    {model:20s} mean(I)={np.mean(vals):+.4f} "
              f"({n_neg}/{len(vals)} negative) "
              f"t={t_stat:.2f} p={p_val:.4f}")

        # Top negative interactions
        sorted_i = sorted(i_vals, key=lambda x: x["I"])
        for item in sorted_i[:3]:
            print(f"      {item['identity']:30s} I={item['I']:+.4f}")


def section_wino_model_comparison(all_models, out_dir):
    """[G-analog] Summary comparison table across models."""
    print(f"\n  --- [G] Model Comparison Summary ---")
    plt = setup_plt()

    rows = []
    for model, data in sorted(all_models.items()):
        ov = data["metrics"].get("overall", data["metrics"])
        if ov.get("overall_accuracy") is None:
            continue
        rows.append({
            "model": model,
            "accuracy": ov.get("overall_accuracy"),
            "disparity": ov.get("accuracy_disparity", 0),
            "most_accurate": ov.get("most_accurate", "—"),
            "least_accurate": ov.get("least_accurate", "—"),
            "source": data["source"],
        })

    if not rows:
        return

    print(f"\n    {'Model':20s} {'Accuracy':10s} {'Disparity':10s} {'Best':20s} {'Worst':20s}")
    print(f"    {'-'*80}")
    for r in sorted(rows, key=lambda x: -(x["accuracy"] or 0)):
        print(f"    {r['model']:20s} {r['accuracy']:.4f}    {r['disparity']:.4f}    "
              f"{r['most_accurate']:20s} {r['least_accurate']:20s}")

    # Bar chart: accuracy + disparity per model
    models = [r["model"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    disps = [r["disparity"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = range(len(models))

    ax1.bar(x, accs, color="#1976d2", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Overall Accuracy")
    ax1.set_title("WinoIdentity: Overall Accuracy by Model")
    ax1.set_ylim(0, 1)

    ax2.bar(x, disps, color="#d32f2f", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Accuracy Disparity (max - min)")
    ax2.set_title("WinoIdentity: Accuracy Disparity by Model")

    plt.tight_layout()
    path = out_dir / "wino_model_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"\n    Saved: {path}")


def section_wino_sae(all_models, sae_dirs, out_dir):
    """Compare WinoIdentity SAE nFEP with accuracy deviation per identity."""
    plt = setup_plt()

    for sae_dir in sae_dirs:
        results_file = sae_dir / "sae_results.json"
        if not results_file.exists():
            continue

        with open(results_file) as f:
            sae_data = json.load(f)

        sae_model = sae_data.get("model", "")
        sae_results = sae_data.get("results", [])
        layer = sae_data.get("layer", "?")

        if not sae_results:
            continue

        # Find matching WinoIdentity behavioral model
        wino_data = None
        for wmodel, wdata in all_models.items():
            if sae_model in wmodel or wmodel in sae_model:
                wino_data = wdata
                break
        if not wino_data:
            print(f"    No WinoIdentity match for SAE model {sae_model}")
            continue

        pi = _get_pi(wino_data)
        if not pi:
            continue
        base_acc = _get_baseline_acc(pi)
        if base_acc is None:
            continue

        # Build identity-level nFEP
        nfep_by_id = defaultdict(list)
        for r in sae_results:
            combined = r.get("combined", "")
            delta_nfep = r.get("delta_nfep", r.get("nfep"))
            if combined and delta_nfep is not None:
                nfep_by_id[combined].append(delta_nfep)

        # Match with WinoIdentity accuracy
        joint = []
        for ident, nfeps in nfep_by_id.items():
            # Try both orderings
            acc_entry = pi.get(ident)
            if not acc_entry or not isinstance(acc_entry, dict):
                parts = ident.split("+")
                if len(parts) == 2:
                    acc_entry = pi.get(f"{parts[1]}+{parts[0]}")
            if not acc_entry or not isinstance(acc_entry, dict) or acc_entry.get("accuracy") is None:
                continue
            joint.append({
                "identity": ident,
                "nfep": float(np.mean(nfeps)),
                "accuracy": acc_entry["accuracy"],
                "acc_dev": acc_entry["accuracy"] - base_acc,
                "abs_acc_dev": abs(acc_entry["accuracy"] - base_acc),
            })

        if len(joint) < 5:
            print(f"    {sae_model} L{layer}: only {len(joint)} matched identities")
            continue

        x = [j["nfep"] for j in joint]
        y = [j["abs_acc_dev"] for j in joint]
        rho, p_val = sp.spearmanr(x, y)

        sig = "✓ significant" if p_val < 0.05 else "not significant"
        print(f"    {sae_model} L{layer}: ρ={rho:.3f}, p={p_val:.4f} (n={len(joint)}) — {sig}")

        sorted_j = sorted(joint, key=lambda x: x["nfep"], reverse=True)
        print(f"      Highest nFEP:")
        for j in sorted_j[:5]:
            print(f"        {j['identity']:25s} nFEP={j['nfep']:.4f}  acc_dev={j['acc_dev']:+.4f}")
        print(f"      Lowest nFEP:")
        for j in sorted_j[-3:]:
            print(f"        {j['identity']:25s} nFEP={j['nfep']:.4f}  acc_dev={j['acc_dev']:+.4f}")

        # Plot
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.scatter(x, y, s=40, alpha=0.6)
        for j in joint:
            if j["nfep"] > np.percentile(x, 85) or j["abs_acc_dev"] > np.percentile(y, 85):
                ax.annotate(j["identity"], (j["nfep"], j["abs_acc_dev"]),
                           fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points")
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(min(x), max(x), 100)
            ax.plot(x_line, np.poly1d(z)(x_line), "--", color="gray", alpha=0.5)
        ax.set_xlabel("Delta nFEP (SAE compositional collapse)")
        ax.set_ylabel("|Accuracy Deviation from Baseline|")
        ax.set_title(f"H2 WinoIdentity: {sae_model} L{layer}\n"
                     f"ρ={rho:.3f}, p={p_val:.4f} (n={len(joint)})")
        plt.tight_layout()
        path = out_dir / f"h2_wino_{sae_model}_L{layer}.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"      Saved: {path}")


if __name__ == "__main__":
    main()