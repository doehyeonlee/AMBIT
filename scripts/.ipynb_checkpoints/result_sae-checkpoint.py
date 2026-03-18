"""
Statistical testing and visualization for SAE compositional collapse results.

Reads: outputs/sae/*/sae_results.json  (from run_sae_analysis.py)
       outputs/sae/*/summary.json

Produces:
  1. nFEP by trait: which traits collapse most (delta vs raw comparison)
  2. nFEP by layer: layer-wise collapse progression
  3. nFEP ↔ behavioral disparity correlation (H2)
  4. Cross-context nFEP: JobFair vs LBOX SAE comparison
  5. Statistical tests: Kruskal-Wallis, Spearman, paired t-test

Usage:
  python -m scripts.result_sae
  python -m scripts.result_sae --output-dir outputs/figures
"""

import json, argparse, sys, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats as sp_module

warnings.filterwarnings("ignore", category=FutureWarning)


def bootstrap_ci(x, y, stat_func=None, n_boot=2000, ci=95):
    """Bootstrap confidence interval for a statistic on (x, y) pairs."""
    if stat_func is None:
        stat_func = lambda a, b: sp_module.spearmanr(a, b)[0]
    x, y = np.array(x), np.array(y)
    boot_stats = []
    for _ in range(n_boot):
        idx = np.random.choice(len(x), len(x), replace=True)
        try:
            boot_stats.append(stat_func(x[idx], y[idx]))
        except Exception:
            pass
    if not boot_stats:
        return (np.nan, np.nan)
    lo = np.percentile(boot_stats, (100 - ci) / 2)
    hi = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    return (round(float(lo), 4), round(float(hi), 4))

PROJECT_ROOT = Path(__file__).parent.parent
SAE_DIR = PROJECT_ROOT / "outputs" / "sae"
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


# ═══════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════

def find_sae_results():
    """Discover all SAE result files."""
    found = []
    if not SAE_DIR.exists():
        return found
    for d in sorted(SAE_DIR.iterdir()):
        if not d.is_dir():
            continue
        results_file = d / "sae_results.json"
        summary_file = d / "summary.json"
        if results_file.exists():
            # Parse dirname: {dataset}_{model}_L{layer}
            parts = d.name.rsplit("_L", 1)
            if len(parts) == 2:
                dataset_model = parts[0]
                layer = int(parts[1])
                # Split dataset_model
                for ds in ["jobfair", "lbox", "winoidentity"]:
                    if dataset_model.startswith(ds + "_"):
                        model = dataset_model[len(ds) + 1:]
                        found.append({
                            "dir": d,
                            "dataset": ds,
                            "model": model,
                            "layer": layer,
                            "results_file": results_file,
                            "summary_file": summary_file if summary_file.exists() else None,
                        })
                        break
    return found


def load_sae_results(results_file):
    """Load individual FEP results from sae_results.json."""
    with open(results_file) as f:
        data = json.load(f)
    return data.get("results", [])


def load_behavioral_summary(dataset):
    """Load behavioral summary for correlation analysis.
    For jobfair/lbox: uses score-based summary.
    For winoidentity: uses accuracy-based winoidentity_summary.json.
    """
    if dataset == "winoidentity":
        # Try winoidentity_summary.json from result_wino.py
        for path in [FIGURE_DIR / "winoidentity_summary.json",
                     ANALYSIS_DIR / "winoidentity" / "summary.json"]:
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        return None
    else:
        path = ANALYSIS_DIR / dataset / "summary.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)


# ═══════════════════════════════════════
# 1. nFEP BY TRAIT
# ═══════════════════════════════════════

def analyze_nfep_by_trait(results, prefix="delta"):
    """
    Group nFEP by individual traits.
    prefix: 'delta' for baseline-subtracted, 'raw' for original.
    """
    nfep_key = f"{prefix}_nfep" if prefix != "nfep" else "nfep"
    by_trait = defaultdict(list)
    for r in results:
        nfep = r.get(nfep_key, r.get("nfep"))
        if nfep is None:
            continue
        by_trait[r["identity_a"]].append(nfep)
        by_trait[r["identity_b"]].append(nfep)

    stats = {}
    for trait, values in sorted(by_trait.items()):
        stats[trait] = {
            "mean": round(float(np.mean(values)), 5),
            "std": round(float(np.std(values)), 5),
            "median": round(float(np.median(values)), 5),
            "n": len(values),
        }
    return stats


def test_trait_differences(results, prefix="delta"):
    """Kruskal-Wallis test: do traits differ in nFEP?"""
    from scipy import stats as sp

    nfep_key = f"{prefix}_nfep" if prefix != "nfep" else "nfep"
    by_trait = defaultdict(list)
    for r in results:
        nfep = r.get(nfep_key, r.get("nfep"))
        if nfep is None:
            continue
        by_trait[r["identity_a"]].append(nfep)
        by_trait[r["identity_b"]].append(nfep)

    groups = [v for v in by_trait.values() if len(v) >= 3]
    if len(groups) < 3:
        return None

    H, p = sp.kruskal(*groups)
    return {"H_statistic": round(float(H), 4), "p_value": float(p), "n_groups": len(groups)}


# ═══════════════════════════════════════
# 2. nFEP BY LAYER
# ═══════════════════════════════════════

def analyze_layer_progression(all_sae, dataset, model):
    """Compare nFEP across layers for same dataset+model."""
    entries = [s for s in all_sae if s["dataset"] == dataset and s["model"] == model]
    if len(entries) < 2:
        return None

    layer_stats = []
    for entry in sorted(entries, key=lambda x: x["layer"]):
        results = load_sae_results(entry["results_file"])
        delta_vals = [r.get("delta_nfep", r.get("nfep", 0)) for r in results if r.get("delta_nfep") is not None or r.get("nfep") is not None]
        raw_vals = [r.get("raw_nfep", 0) for r in results if r.get("raw_nfep") is not None]

        layer_stats.append({
            "layer": entry["layer"],
            "delta_nfep_mean": round(float(np.mean(delta_vals)), 5) if delta_vals else None,
            "delta_nfep_std": round(float(np.std(delta_vals)), 5) if delta_vals else None,
            "raw_nfep_mean": round(float(np.mean(raw_vals)), 5) if raw_vals else None,
            "raw_nfep_std": round(float(np.std(raw_vals)), 5) if raw_vals else None,
            "n": len(delta_vals),
        })

    return layer_stats


def test_layer_trend(layer_stats):
    """Test monotonic increase of nFEP with layer depth."""
    from scipy import stats as sp

    if not layer_stats or len(layer_stats) < 2:
        return None

    layers = [s["layer"] for s in layer_stats]
    nfeps = [s["delta_nfep_mean"] for s in layer_stats if s["delta_nfep_mean"] is not None]

    if len(nfeps) < 2:
        return None

    rho, p = sp.spearmanr(layers[:len(nfeps)], nfeps)
    return {"spearman_rho": round(float(rho), 4), "p_value": float(p), "n_layers": len(nfeps)}


# ═══════════════════════════════════════
# 3. nFEP ↔ BEHAVIORAL DISPARITY (H2)
# ═══════════════════════════════════════

def correlate_nfep_behavioral(results, behavioral_summary, model_key, dataset=""):
    """
    Spearman correlation: trait nFEP ↔ behavioral disparity.
    For jobfair/lbox: uses |score deviation|.
    For winoidentity: uses |accuracy deviation|.
    Tests H2: SAE collapse predicts behavioral bias.
    """
    from scipy import stats as sp

    if model_key not in behavioral_summary:
        return None

    model_data = behavioral_summary[model_key]

    # Determine metric type
    if dataset == "winoidentity":
        # WinoIdentity: use accuracy from metrics
        pi = model_data.get("metrics", model_data)
        if "per_identity" in pi:
            pi = pi["per_identity"]
        # Get baseline accuracy (or mean)
        base_val = pi.get("(baseline)", {}).get("accuracy")
        if base_val is None:
            accs = [v["accuracy"] for v in pi.values()
                    if isinstance(v, dict) and v.get("accuracy") is not None]
            base_val = float(np.mean(accs)) if accs else None
        if base_val is None:
            return None
        # Trait-level accuracy deviation
        behavioral_dev = {}
        for key, stat in pi.items():
            if not isinstance(stat, dict) or stat.get("accuracy") is None or key == "(baseline)":
                continue
            parts = key.split("+")
            for p in parts:
                if p not in behavioral_dev:
                    behavioral_dev[p] = []
                behavioral_dev[p].append(abs(stat["accuracy"] - base_val))
    else:
        # JobFair/LBOX: use score
        pi = model_data.get("per_identity", model_data)
        bl_mean = pi.get("(baseline)", {}).get("mean_score")
        if bl_mean is None:
            return None
        behavioral_dev = {}
        for key, stat in pi.items():
            if stat.get("mean_score") is None or key == "(baseline)":
                continue
            parts = key.split("+")
            for p in parts:
                if p not in behavioral_dev:
                    behavioral_dev[p] = []
                behavioral_dev[p].append(abs(stat["mean_score"] - bl_mean))

    behavioral_trait_dev = {t: np.mean(vs) for t, vs in behavioral_dev.items() if len(vs) >= 2}

    # Trait-level nFEP
    nfep_by_trait = analyze_nfep_by_trait(results, "delta")

    # Common traits
    common = sorted(set(nfep_by_trait.keys()) & set(behavioral_trait_dev.keys()))
    if len(common) < 5:
        return None

    x_nfep = [nfep_by_trait[t]["mean"] for t in common]
    y_dev = [behavioral_trait_dev[t] for t in common]

    rho, p = sp.spearmanr(x_nfep, y_dev)
    ci_lo, ci_hi = bootstrap_ci(x_nfep, y_dev)
    return {
        "spearman_rho": round(float(rho), 4),
        "p_value": float(p),
        "ci_95": [ci_lo, ci_hi],
        "n_traits": len(common),
        "metric_type": "accuracy" if dataset == "winoidentity" else "score",
        "traits": [{"trait": t, "nfep": round(nfep_by_trait[t]["mean"], 5),
                     "behavioral_dev": round(behavioral_trait_dev[t], 4)} for t in common],
    }


# ═══════════════════════════════════════
# 4. CROSS-CONTEXT SAE
# ═══════════════════════════════════════

def cross_context_nfep(all_sae, model):
    """Compare nFEP patterns across datasets for same model."""
    from scipy import stats as sp

    datasets = defaultdict(list)
    for entry in all_sae:
        if entry["model"] == model:
            results = load_sae_results(entry["results_file"])
            trait_stats = analyze_nfep_by_trait(results, "delta")
            datasets[entry["dataset"]].append({"layer": entry["layer"], "traits": trait_stats})

    ds_names = sorted(datasets.keys())
    if len(ds_names) < 2:
        return None

    # Compare trait ordering between first two datasets
    ds1, ds2 = ds_names[0], ds_names[1]
    # Use last layer for each
    t1 = datasets[ds1][-1]["traits"] if datasets[ds1] else {}
    t2 = datasets[ds2][-1]["traits"] if datasets[ds2] else {}

    common = sorted(set(t1.keys()) & set(t2.keys()))
    if len(common) < 5:
        return None

    x = [t1[t]["mean"] for t in common]
    y = [t2[t]["mean"] for t in common]
    rho, p = sp.spearmanr(x, y)

    return {
        "model": model,
        "dataset_1": ds1, "dataset_2": ds2,
        "spearman_rho": round(float(rho), 4),
        "p_value": float(p),
        "n_traits": len(common),
    }


# ═══════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════

def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
    return plt


def plot_nfep_by_trait(trait_stats_delta, trait_stats_raw, dataset, model, layer, out_dir):
    """Side-by-side bar chart: delta vs raw nFEP by trait."""
    plt = setup_matplotlib()

    traits = sorted(trait_stats_delta.keys(), key=lambda t: trait_stats_delta[t]["mean"], reverse=True)
    delta_vals = [trait_stats_delta[t]["mean"] for t in traits]
    raw_vals = [trait_stats_raw.get(t, {}).get("mean", 0) for t in traits]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Delta
    colors = ["#d32f2f" if v > np.mean(delta_vals) else "#1976d2" for v in delta_vals]
    ax1.barh(range(len(traits)), delta_vals, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(traits)))
    ax1.set_yticklabels(traits, fontsize=9)
    ax1.set_xlabel("Delta nFEP (baseline-subtracted)")
    ax1.set_title(f"Delta nFEP: {model} L{layer} on {dataset}")
    ax1.invert_yaxis()

    # Raw
    colors_raw = ["#d32f2f" if v > np.mean(raw_vals) else "#1976d2" for v in raw_vals]
    ax2.barh(range(len(traits)), raw_vals, color=colors_raw, alpha=0.8)
    ax2.set_yticks(range(len(traits)))
    ax2.set_yticklabels(traits, fontsize=9)
    ax2.set_xlabel("Raw nFEP")
    ax2.set_title(f"Raw nFEP: {model} L{layer} on {dataset}")
    ax2.invert_yaxis()

    plt.tight_layout()
    path = out_dir / f"nfep_traits_{dataset}_{model}_L{layer}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_layer_progression(layer_stats, dataset, model, out_dir):
    """Line plot: nFEP across layers (delta + raw)."""
    plt = setup_matplotlib()

    layers = [s["layer"] for s in layer_stats]
    delta_means = [s["delta_nfep_mean"] or 0 for s in layer_stats]
    delta_stds = [s["delta_nfep_std"] or 0 for s in layer_stats]
    raw_means = [s["raw_nfep_mean"] or 0 for s in layer_stats]
    raw_stds = [s["raw_nfep_std"] or 0 for s in layer_stats]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(layers, delta_means, yerr=delta_stds, marker="o", linewidth=2, label="Delta nFEP", capsize=5)
    ax.errorbar(layers, raw_means, yerr=raw_stds, marker="s", linewidth=2, label="Raw nFEP", capsize=5, alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("nFEP")
    ax.set_title(f"Layer-wise Compositional Collapse: {model} on {dataset}")
    ax.legend()
    plt.tight_layout()

    path = out_dir / f"nfep_layers_{dataset}_{model}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_nfep_vs_behavioral(corr_result, dataset, model, out_dir):
    """Scatter: nFEP vs behavioral deviation per trait."""
    plt = setup_matplotlib()

    if not corr_result or "traits" not in corr_result:
        return

    traits = corr_result["traits"]
    x = [t["nfep"] for t in traits]
    y = [t["behavioral_dev"] for t in traits]
    labels = [t["trait"] for t in traits]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x, y, s=50, alpha=0.7)
    for i, label in enumerate(labels):
        ax.annotate(label, (x[i], y[i]), fontsize=8, alpha=0.8,
                   xytext=(5, 5), textcoords="offset points")

    # Trend line
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.5)

    ax.set_xlabel("Delta nFEP (SAE compositional collapse)")
    ax.set_ylabel("Behavioral |Score Deviation|")
    ax.set_title(f"H2: SAE Collapse ↔ Behavioral Bias\n{model} on {dataset} "
                 f"(ρ={corr_result['spearman_rho']:.3f}, p={corr_result['p_value']:.4f})")
    plt.tight_layout()

    path = out_dir / f"h2_nfep_vs_behavioral_{dataset}_{model}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="SAE nFEP analysis and visualization")
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover SAE results
    all_sae = find_sae_results()
    if not all_sae:
        print("No SAE results found in outputs/sae/. Run run_sae_analysis first.")
        sys.exit(1)

    print(f"Found {len(all_sae)} SAE result sets:")
    for s in all_sae:
        print(f"  {s['dataset']}/{s['model']}/L{s['layer']}")

    all_stats = {}

    # ── 1. nFEP by trait ──
    print("\n=== 1. nFEP by Trait ===")
    for entry in all_sae:
        results = load_sae_results(entry["results_file"])
        if not results:
            continue

        ds, model, layer = entry["dataset"], entry["model"], entry["layer"]
        print(f"\n  [{ds}/{model}/L{layer}] ({len(results)} groups)")

        delta_stats = analyze_nfep_by_trait(results, "delta")
        raw_stats = analyze_nfep_by_trait(results, "raw")

        if delta_stats:
            sorted_d = sorted(delta_stats.items(), key=lambda x: x[1]["mean"], reverse=True)
            print(f"    Delta nFEP — top 5:")
            for t, s in sorted_d[:5]:
                raw_v = raw_stats.get(t, {}).get("mean", 0)
                print(f"      {t:25s} delta={s['mean']:.5f}  raw={raw_v:.5f}")
            print(f"    Delta nFEP — bottom 3:")
            for t, s in sorted_d[-3:]:
                raw_v = raw_stats.get(t, {}).get("mean", 0)
                print(f"      {t:25s} delta={s['mean']:.5f}  raw={raw_v:.5f}")

        # Kruskal-Wallis
        kw = test_trait_differences(results, "delta")
        if kw:
            sig = "significant" if kw["p_value"] < 0.05 else "not significant"
            print(f"    Kruskal-Wallis: H={kw['H_statistic']:.2f}, p={kw['p_value']:.4f} ({sig})")

        plot_nfep_by_trait(delta_stats, raw_stats, ds, model, layer, out_dir)
        all_stats[f"trait_{ds}_{model}_L{layer}"] = {"delta": delta_stats, "raw": raw_stats, "kruskal_wallis": kw}

    # ── 2. Layer progression ──
    print("\n=== 2. Layer Progression ===")
    seen = set()
    for entry in all_sae:
        key = (entry["dataset"], entry["model"])
        if key in seen:
            continue
        seen.add(key)

        layer_stats = analyze_layer_progression(all_sae, entry["dataset"], entry["model"])
        if not layer_stats or len(layer_stats) < 2:
            continue

        ds, model = entry["dataset"], entry["model"]
        print(f"\n  [{ds}/{model}] {len(layer_stats)} layers")
        for ls in layer_stats:
            print(f"    L{ls['layer']:3d}: delta={ls['delta_nfep_mean']:.5f}  raw={ls['raw_nfep_mean']:.5f}  (n={ls['n']})")

        trend = test_layer_trend(layer_stats)
        if trend:
            direction = "increasing" if trend["spearman_rho"] > 0 else "decreasing"
            print(f"    Trend: ρ={trend['spearman_rho']:.3f}, p={trend['p_value']:.4f} ({direction})")

        plot_layer_progression(layer_stats, ds, model, out_dir)
        all_stats[f"layers_{ds}_{model}"] = {"stats": layer_stats, "trend": trend}

    # ── 3. nFEP ↔ Behavioral correlation (H2) ──
    print("\n=== 3. nFEP ↔ Behavioral Disparity (H2) ===")

    # Use deepest layer per dataset/model for H2
    deepest = {}
    for entry in all_sae:
        key = (entry["dataset"], entry["model"])
        if key not in deepest or entry["layer"] > deepest[key]["layer"]:
            deepest[key] = entry

    for (ds, model), entry in sorted(deepest.items()):
        results = load_sae_results(entry["results_file"])
        if not results:
            continue

        behavioral = load_behavioral_summary(ds)
        if not behavioral:
            print(f"  [{ds}/{model}] No behavioral summary found")
            continue

        # Robust model matching: try exact, substring, and common prefix
        bm_key = None
        for k in behavioral.keys():
            if k == model or model in k or k in model:
                bm_key = k
                break
        # Also try with hyphens replaced
        if not bm_key:
            for k in behavioral.keys():
                if k.replace("-", "") == model.replace("-", ""):
                    bm_key = k
                    break

        if not bm_key:
            # List available keys for debugging
            print(f"  [{ds}/{model}] No behavioral match. Available keys: {list(behavioral.keys())[:5]}")
            continue

        corr = correlate_nfep_behavioral(results, behavioral, bm_key, dataset=ds)
        if corr:
            sig = "H2 supported" if corr["p_value"] < 0.05 and corr["spearman_rho"] > 0 else "H2 not supported"
            print(f"  [{ds}/{model} L{entry['layer']}] ↔ behavioral [{bm_key}]:")
            ci = corr.get('ci_95', [None, None])
            ci_str = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else ""
            print(f"    ρ={corr['spearman_rho']:.3f}{ci_str}, p={corr['p_value']:.4f} → {sig}")

            # Print per-trait detail
            if "traits" in corr:
                sorted_traits = sorted(corr["traits"], key=lambda x: x["nfep"], reverse=True)
                print(f"    Trait-level (sorted by nFEP):")
                for t in sorted_traits[:5]:
                    print(f"      {t['trait']:25s} nFEP={t['nfep']:.5f}  |behav_dev|={t['behavioral_dev']:.4f}")
                print(f"      ...")
                for t in sorted_traits[-3:]:
                    print(f"      {t['trait']:25s} nFEP={t['nfep']:.5f}  |behav_dev|={t['behavioral_dev']:.4f}")

            plot_nfep_vs_behavioral(corr, ds, model, out_dir)
            all_stats[f"h2_{ds}_{model}_L{entry['layer']}"] = corr
        else:
            print(f"  [{ds}/{model}] Correlation failed — check data overlap")

    # ── 3b. Identity-level SAE ↔ Behavioral joint table ──
    print("\n=== 3b. Identity-Level SAE ↔ Behavioral Joint Analysis ===")
    for (ds, model), entry in sorted(deepest.items()):
        results = load_sae_results(entry["results_file"])
        behavioral = load_behavioral_summary(ds)
        if not results or not behavioral:
            continue

        bm_key = None
        for k in behavioral.keys():
            if k == model or model in k or k in model:
                bm_key = k
                break
        if not bm_key:
            continue

        # Determine data format: scoring (mean_score) or winoidentity (accuracy)
        model_beh = behavioral[bm_key]
        if ds == "winoidentity":
            # WinoIdentity: metrics → per_identity → accuracy
            pi = model_beh.get("metrics", model_beh)
            if "per_identity" in pi:
                pi = pi["per_identity"]
            # Get baseline
            base_val = pi.get("(baseline)", {}).get("accuracy")
            if base_val is None:
                accs = [v["accuracy"] for v in pi.values()
                        if isinstance(v, dict) and v.get("accuracy") is not None]
                base_val = float(np.mean(accs)) if accs else None
            if base_val is None:
                continue
            metric_key = "accuracy"
        else:
            pi = model_beh.get("per_identity", {})
            base_val = pi.get("(baseline)", {}).get("mean_score")
            if base_val is None:
                continue
            metric_key = "mean_score"

        # Build per-identity nFEP (using "combined" field = "gender+trait")
        nfep_by_id = defaultdict(list)
        for r in results:
            combined = r.get("combined", "")
            delta_nfep = r.get("delta_nfep", r.get("nfep"))
            if combined and delta_nfep is not None:
                nfep_by_id[combined].append(delta_nfep)

        # Match with behavioral
        joint = []
        for ident, nfeps in nfep_by_id.items():
            behav = pi.get(ident)
            if not behav or not isinstance(behav, dict):
                # Try swapped order
                parts = ident.split("+")
                if len(parts) == 2:
                    behav = pi.get(f"{parts[1]}+{parts[0]}")
            if not behav or not isinstance(behav, dict):
                continue
            val = behav.get(metric_key)
            if val is None:
                continue
            dev = val - base_val
            # LBOX: higher score = more severe = worse → flip so positive = favorable
            if ds == "lbox":
                dev = -dev
            joint.append({
                "identity": ident,
                "nfep": float(np.mean(nfeps)),
                "behav_value": val,
                "behav_dev": dev,
                "behav_abs_dev": abs(val - base_val),
            })

        if len(joint) < 5:
            continue

        # Spearman at identity level
        from scipy import stats as sp_stats
        x = [j["nfep"] for j in joint]
        y = [j["behav_abs_dev"] for j in joint]
        rho, p_val = sp_stats.spearmanr(x, y)
        ci_lo, ci_hi = bootstrap_ci(x, y)

        print(f"\n  [{ds}/{model} L{entry['layer']}] Identity-level correlation:")
        print(f"    ρ={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p={p_val:.4f} (n={len(joint)} identities)")

        # Top/bottom
        sorted_j = sorted(joint, key=lambda x: x["nfep"], reverse=True)
        metric_label = "acc_dev" if ds == "winoidentity" else "score_dev"
        dev_note = "(LBOX: negative = more lenient = favorable)" if ds == "lbox" else \
                   "(WinoIdentity: accuracy deviation)" if ds == "winoidentity" else ""
        print(f"    Highest nFEP identities: {dev_note}")
        for j in sorted_j[:5]:
            print(f"      {j['identity']:30s} nFEP={j['nfep']:.4f}  {metric_label}={j['behav_dev']:+.3f}")
        print(f"    Lowest nFEP identities:")
        for j in sorted_j[-5:]:
            print(f"      {j['identity']:30s} nFEP={j['nfep']:.4f}  {metric_label}={j['behav_dev']:+.3f}")

        # Plot identity-level scatter
        plt = setup_matplotlib()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(x, y, s=30, alpha=0.6)
        for j in joint:
            if j["nfep"] > np.percentile(x, 90) or j["behav_abs_dev"] > np.percentile(y, 90):
                ax.annotate(j["identity"], (j["nfep"], j["behav_abs_dev"]),
                           fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(min(x), max(x), 100)
            ax.plot(x_line, np.poly1d(z)(x_line), "--", color="gray", alpha=0.5)
        ax.set_xlabel("Delta nFEP (SAE compositional collapse)")
        y_label = "|Accuracy Deviation|" if ds == "winoidentity" else "|Behavioral Score Deviation|"
        ax.set_ylabel(y_label)
        ax.set_title(f"H2 Identity-Level: {model} L{entry['layer']} on {ds}\n"
                     f"ρ={rho:.3f}, p={p_val:.4f} (n={len(joint)})")
        plt.tight_layout()
        path = out_dir / f"h2_identity_{ds}_{model}_L{entry['layer']}.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"    Saved: {path}")

        all_stats[f"h2_identity_{ds}_{model}_L{entry['layer']}"] = {
            "spearman_rho": round(float(rho), 4), "p_value": float(p_val),
            "ci_95": [ci_lo, ci_hi],
            "n_identities": len(joint),
            "top5": sorted_j[:5], "bottom5": sorted_j[-5:],
        }

    # ── 4. Cross-context SAE ──
    print("\n=== 4. Cross-Context SAE Comparison ===")
    models_with_sae = set(s["model"] for s in all_sae)
    for model in sorted(models_with_sae):
        cc = cross_context_nfep(all_sae, model)
        if cc:
            print(f"  {model}: {cc['dataset_1']} ↔ {cc['dataset_2']}")
            print(f"    ρ={cc['spearman_rho']:.3f}, p={cc['p_value']:.4f} "
                  f"({'consistent' if cc['spearman_rho'] > 0.5 else 'divergent'} trait ordering)")
            all_stats[f"cross_sae_{model}"] = cc

    # Save
    stats_path = out_dir / "sae_statistical_tests.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\n  Stats saved: {stats_path}")
    print(f"  Figures in: {out_dir}")


if __name__ == "__main__":
    main()