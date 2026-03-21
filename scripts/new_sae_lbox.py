"""
LBOX-Focused SAE Analysis: Why does SAE predict behavioral bias selectively?

RQ3: Do SAE compositional metrics predict behavioral bias selectively
     based on task-relevance of identity information?

Finding: ΔnFEP → behavioral deviation is significant only in LBOX.
  LBOX:   Gemma ρ=+0.292 (p=.040), LLaMA ρ=+0.674 (p<.001)
  JobFair: ρ=0.055 (n.s.)
  Mind:    ρ=−0.012 ~ +0.282 (n.s.)
  Wino:    12/12 n.s.

This script provides two additional analyses to strengthen this finding:

Direction 1: SAE Feature Overlap Analysis
  - Compare ACTUAL feature activation vectors across contexts
  - Jaccard similarity of top-k activated features per identity
  - Distinguish "different features" vs "same features, different strength"

Direction 2: Mediation Analysis
  - Test whether ΔnFEP MEDIATES the relationship context → behavioral bias
  - Or whether context affects both independently (confound)

Usage: python -m scripts.new_sae_lbox [--output-dir outputs/figures]
"""

import json, argparse, sys, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats as sp

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
SAE_DIR = PROJECT_ROOT / "outputs" / "sae"
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

FLIP_DOMAINS = {"lbox", "mind"}


def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
    return plt


def find_sae_results():
    """Discover all SAE result files."""
    found = []
    if not SAE_DIR.exists():
        return found
    for d in sorted(SAE_DIR.iterdir()):
        if not d.is_dir():
            continue
        results_file = d / "sae_results.json"
        if results_file.exists():
            parts = d.name.rsplit("_L", 1)
            if len(parts) == 2:
                dataset_model = parts[0]
                layer = int(parts[1])
                for ds in ["jobfair", "lbox", "mind", "winoidentity"]:
                    if dataset_model.startswith(ds + "_"):
                        model = dataset_model[len(ds) + 1:]
                        found.append({
                            "dir": d, "dataset": ds, "model": model, "layer": layer,
                            "results_file": results_file,
                        })
                        break
    return found


def load_sae_results(results_file):
    with open(results_file) as f:
        data = json.load(f)
    return data.get("results", [])


def load_behavioral_summary(dataset):
    if dataset == "winoidentity":
        for path in [FIGURE_DIR / "winoidentity_summary.json",
                     ANALYSIS_DIR / "winoidentity" / "summary.json"]:
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        return None
    path = ANALYSIS_DIR / dataset / "summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def bootstrap_ci(x, y, stat_func=None, n_boot=2000, ci=95):
    if stat_func is None:
        stat_func = lambda a, b: sp.spearmanr(a, b)[0]
    x, y = np.array(x), np.array(y)
    boot = []
    for _ in range(n_boot):
        idx = np.random.choice(len(x), len(x), replace=True)
        try:
            boot.append(stat_func(x[idx], y[idx]))
        except:
            pass
    if not boot:
        return (np.nan, np.nan)
    return (round(float(np.percentile(boot, (100-ci)/2)), 4),
            round(float(np.percentile(boot, 100-(100-ci)/2)), 4))


# ═══════════════════════════════════════════════════════════
# DIRECTION 1: Feature Overlap Analysis
# ═══════════════════════════════════════════════════════════

def direction1_feature_overlap(all_sae, out_dir):
    """
    Compare SAE feature activation patterns across contexts for same model.

    For each identity (e.g., "male+fat"):
      - In LBOX: which SAE features are in the top-k residual?
      - In JobFair: which features?
      - Jaccard similarity = |A ∩ B| / |A ∪ B|

    If Jaccard ≈ 0: completely different features → different mechanisms
    If Jaccard ≈ 1: same features → same mechanism, different magnitude
    """
    print(f"\n{'='*70}")
    print("DIRECTION 1: SAE Feature Overlap Analysis")
    print(f"{'='*70}")

    plt = setup_plt()

    # Group SAE results by (model, layer)
    by_model_layer = defaultdict(dict)
    for entry in all_sae:
        key = (entry["model"], entry["layer"])
        by_model_layer[key][entry["dataset"]] = entry

    results = {}

    for (model, layer), ds_entries in sorted(by_model_layer.items()):
        datasets = sorted(ds_entries.keys())
        if len(datasets) < 2:
            continue

        print(f"\n  [{model} L{layer}] Contexts: {datasets}")

        # Load results for each dataset
        ds_results = {}
        for ds in datasets:
            ds_results[ds] = load_sae_results(ds_entries[ds]["results_file"])

        # Build per-identity feature sets for each dataset
        # Each result has "delta_residual_top_features" (top-20 feature indices)
        ds_features = {}
        for ds, res_list in ds_results.items():
            by_identity = defaultdict(list)
            for r in res_list:
                combined = r.get("combined", "")
                top_feats = r.get("delta_residual_top_features", r.get("residual_top_features", []))
                if combined and top_feats:
                    by_identity[combined].append(set(top_feats[:10]))  # top-10 per group
            # Merge across groups for same identity: union of all top features
            ds_features[ds] = {}
            for ident, feat_sets in by_identity.items():
                merged = set()
                for fs in feat_sets:
                    merged.update(fs)
                ds_features[ds][ident] = merged

        # Pairwise Jaccard similarity between contexts
        for i, ds_a in enumerate(datasets):
            for ds_b in datasets[i+1:]:
                common_ids = sorted(set(ds_features[ds_a].keys()) & set(ds_features[ds_b].keys()))
                if not common_ids:
                    continue

                jaccards = []
                overlap_details = []
                for ident in common_ids:
                    fa = ds_features[ds_a][ident]
                    fb = ds_features[ds_b][ident]
                    intersection = len(fa & fb)
                    union = len(fa | fb)
                    j = intersection / union if union > 0 else 0
                    jaccards.append(j)
                    overlap_details.append({
                        "identity": ident, "jaccard": round(j, 4),
                        "n_a": len(fa), "n_b": len(fb), "n_shared": intersection,
                    })

                mean_j = np.mean(jaccards)
                std_j = np.std(jaccards)

                print(f"\n    {ds_a} ↔ {ds_b}: mean Jaccard={mean_j:.4f} ± {std_j:.4f} (n={len(common_ids)} identities)")

                # Highest and lowest overlap
                sorted_details = sorted(overlap_details, key=lambda x: x["jaccard"], reverse=True)
                print(f"      Highest overlap:")
                for d in sorted_details[:3]:
                    print(f"        {d['identity']:25s} J={d['jaccard']:.4f} shared={d['n_shared']}/{d['n_a']}+{d['n_b']}")
                print(f"      Lowest overlap:")
                for d in sorted_details[-3:]:
                    print(f"        {d['identity']:25s} J={d['jaccard']:.4f} shared={d['n_shared']}/{d['n_a']}+{d['n_b']}")

                # Key question: does overlap predict behavioral correlation?
                # Identities with higher feature overlap between LBOX↔JobFair
                # should have more consistent bias across contexts
                key = f"{model}_L{layer}_{ds_a}_{ds_b}"
                results[key] = {
                    "model": model, "layer": layer,
                    "dataset_a": ds_a, "dataset_b": ds_b,
                    "mean_jaccard": round(mean_j, 4), "std_jaccard": round(std_j, 4),
                    "n_identities": len(common_ids),
                    "details": sorted_details,
                }

        # Cross-context feature overlap heatmap
        if len(datasets) >= 2:
            n_ds = len(datasets)
            overlap_matrix = np.zeros((n_ds, n_ds))
            for i in range(n_ds):
                for j in range(n_ds):
                    if i == j:
                        overlap_matrix[i, j] = 1.0
                        continue
                    common_ids = set(ds_features[datasets[i]].keys()) & set(ds_features[datasets[j]].keys())
                    if common_ids:
                        jacs = []
                        for ident in common_ids:
                            fa = ds_features[datasets[i]][ident]
                            fb = ds_features[datasets[j]][ident]
                            union = len(fa | fb)
                            jacs.append(len(fa & fb) / union if union > 0 else 0)
                        overlap_matrix[i, j] = np.mean(jacs)

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(overlap_matrix, cmap="YlOrRd", vmin=0, vmax=0.5)
            ax.set_xticks(range(n_ds))
            ax.set_xticklabels(datasets, rotation=45, ha="right")
            ax.set_yticks(range(n_ds))
            ax.set_yticklabels(datasets)
            for i in range(n_ds):
                for j in range(n_ds):
                    ax.text(j, i, f"{overlap_matrix[i,j]:.3f}", ha="center", va="center", fontsize=10)
            ax.set_title(f"Feature Overlap (Jaccard): {model} L{layer}")
            plt.colorbar(im, ax=ax, shrink=0.7, label="Mean Jaccard Similarity")
            plt.tight_layout()
            path = out_dir / f"feature_overlap_{model}_L{layer}.png"
            fig.savefig(path)
            plt.close(fig)
            print(f"\n    Saved: {path}")

    return results


# ═══════════════════════════════════════════════════════════
# DIRECTION 2: Mediation Analysis
# ═══════════════════════════════════════════════════════════

def direction2_mediation(all_sae, out_dir):
    """
    Mediation analysis: Does ΔnFEP mediate the context → behavioral bias relationship?

    Model:
      Context (LBOX vs JobFair) → ΔnFEP → Behavioral Deviation

    Steps (Baron & Kenny 1986):
      Path c:  Context → Behavioral (total effect)
      Path a:  Context → ΔnFEP (does context affect internal representation?)
      Path b:  ΔnFEP → Behavioral (controlling for context)
      Path c': Context → Behavioral (controlling for ΔnFEP = direct effect)

    Mediation = c - c' (indirect effect through ΔnFEP)
    Sobel test for significance.
    """
    print(f"\n{'='*70}")
    print("DIRECTION 2: Mediation Analysis (Context → ΔnFEP → Behavioral)")
    print(f"{'='*70}")

    plt = setup_plt()

    # Group by model
    by_model = defaultdict(dict)
    for entry in all_sae:
        if entry["dataset"] in ["jobfair", "lbox", "mind"]:
            key = (entry["model"], entry["layer"])
            by_model[key][entry["dataset"]] = entry

    results = {}

    for (model, layer), ds_entries in sorted(by_model.items()):
        scoring_datasets = sorted(ds_entries.keys())
        if len(scoring_datasets) < 2:
            continue

        print(f"\n  [{model} L{layer}] Scoring contexts: {scoring_datasets}")

        # Load SAE nFEP per identity for each dataset
        nfep_by_ds = {}
        for ds in scoring_datasets:
            res_list = load_sae_results(ds_entries[ds]["results_file"])
            by_id = defaultdict(list)
            for r in res_list:
                by_id[r["combined"]].append(r["nfep"])
            nfep_by_ds[ds] = {k: np.mean(v) for k, v in by_id.items()}

        # Load behavioral deviation per identity for each dataset
        behav_by_ds = {}
        for ds in scoring_datasets:
            summary = load_behavioral_summary(ds)
            if not summary:
                continue
            # Find model key
            bm_key = None
            for k in summary.keys():
                if k == model or model in k or k in model:
                    bm_key = k
                    break
            if not bm_key:
                for k in summary.keys():
                    if k.replace("-", "") == model.replace("-", ""):
                        bm_key = k
                        break
            if not bm_key:
                continue

            pi = summary[bm_key].get("per_identity", {})
            bl = pi.get("(baseline)", {}).get("mean_score")
            if bl is None:
                continue

            devs = {}
            for ident, stats in pi.items():
                if stats.get("mean_score") is not None and ident != "(baseline)":
                    dev = stats["mean_score"] - bl
                    if ds in FLIP_DOMAINS:
                        dev = -dev
                    devs[ident] = dev
            behav_by_ds[ds] = devs

        # Mediation for each pair of contexts
        for i, ds_a in enumerate(scoring_datasets):
            for ds_b in scoring_datasets[i+1:]:
                if ds_a not in nfep_by_ds or ds_b not in nfep_by_ds:
                    continue
                if ds_a not in behav_by_ds or ds_b not in behav_by_ds:
                    continue

                # Build joint dataset: each identity × 2 contexts
                common = sorted(
                    set(nfep_by_ds[ds_a].keys()) & set(nfep_by_ds[ds_b].keys()) &
                    set(behav_by_ds[ds_a].keys()) & set(behav_by_ds[ds_b].keys())
                )
                if len(common) < 10:
                    print(f"    {ds_a} ↔ {ds_b}: only {len(common)} common identities, skipping")
                    continue

                # Construct arrays: context=0/1, nfep, behavioral_dev
                contexts = []
                nfeps = []
                behavs = []
                identities = []

                for ident in common:
                    # Context A
                    contexts.append(0)
                    nfeps.append(nfep_by_ds[ds_a][ident])
                    behavs.append(abs(behav_by_ds[ds_a][ident]))
                    identities.append(ident)
                    # Context B
                    contexts.append(1)
                    nfeps.append(nfep_by_ds[ds_b][ident])
                    behavs.append(abs(behav_by_ds[ds_b][ident]))
                    identities.append(ident)

                contexts = np.array(contexts, dtype=float)
                nfeps = np.array(nfeps)
                behavs = np.array(behavs)

                # Baron & Kenny steps
                # Path c: Context → |Behavioral| (total effect)
                rho_c, p_c = sp.spearmanr(contexts, behavs)

                # Path a: Context → ΔnFEP
                rho_a, p_a = sp.spearmanr(contexts, nfeps)

                # Path b: ΔnFEP → |Behavioral| (across both contexts)
                rho_b, p_b = sp.spearmanr(nfeps, behavs)

                # Partial correlation: Context → |Behavioral| controlling for ΔnFEP
                # Using linear regression residuals
                from numpy.polynomial.polynomial import polyfit
                # Regress behavs on nfeps
                if np.std(nfeps) > 1e-10:
                    coef_bn = np.polyfit(nfeps, behavs, 1)
                    resid_b = behavs - np.polyval(coef_bn, nfeps)
                    coef_cn = np.polyfit(nfeps, contexts, 1)
                    resid_c = contexts - np.polyval(coef_cn, nfeps)
                    if np.std(resid_b) > 1e-10 and np.std(resid_c) > 1e-10:
                        rho_cp, p_cp = sp.spearmanr(resid_c, resid_b)
                    else:
                        rho_cp, p_cp = 0, 1
                else:
                    rho_cp, p_cp = rho_c, p_c

                # Mediation = c - c' (change in total effect after controlling for mediator)
                mediation = rho_c - rho_cp

                # Sobel test (approximate)
                n = len(contexts)
                se_a = np.sqrt((1 - rho_a**2) / (n - 2)) if n > 2 else 1
                se_b = np.sqrt((1 - rho_b**2) / (n - 2)) if n > 2 else 1
                sobel_se = np.sqrt(rho_a**2 * se_b**2 + rho_b**2 * se_a**2)
                sobel_z = (rho_a * rho_b) / sobel_se if sobel_se > 0 else 0
                sobel_p = 2 * (1 - sp.norm.cdf(abs(sobel_z)))

                # Proportion mediated
                prop_mediated = mediation / rho_c if abs(rho_c) > 0.01 else 0

                print(f"\n    {ds_a} ↔ {ds_b} (n={len(common)} identities × 2 = {n} obs)")
                print(f"      Path c  (Context → |Behav|):     ρ={rho_c:+.3f} p={p_c:.4f}")
                print(f"      Path a  (Context → ΔnFEP):      ρ={rho_a:+.3f} p={p_a:.4f}")
                print(f"      Path b  (ΔnFEP → |Behav|):      ρ={rho_b:+.3f} p={p_b:.4f}")
                print(f"      Path c' (Context → |Behav| | M): ρ={rho_cp:+.3f} p={p_cp:.4f}")
                print(f"      Mediation (c - c'): {mediation:+.3f}")
                print(f"      Proportion mediated: {prop_mediated:.1%}")
                print(f"      Sobel test: z={sobel_z:.2f} p={sobel_p:.4f}")

                if abs(rho_a) > 0.1 and p_a < 0.1 and abs(rho_b) > 0.1 and p_b < 0.1:
                    if abs(mediation) > 0.05:
                        if abs(rho_cp) < abs(rho_c) * 0.5:
                            verdict = "Full mediation: ΔnFEP fully mediates context→bias"
                        else:
                            verdict = "Partial mediation: ΔnFEP partially mediates"
                    else:
                        verdict = "No mediation: paths exist but no indirect effect"
                elif abs(rho_b) < 0.1:
                    verdict = "No path b: ΔnFEP does not predict behavior in this pair"
                else:
                    verdict = "Insufficient evidence for mediation"

                print(f"      → {verdict}")

                key = f"{model}_L{layer}_{ds_a}_{ds_b}"
                results[key] = {
                    "model": model, "layer": layer,
                    "dataset_a": ds_a, "dataset_b": ds_b,
                    "n_identities": len(common),
                    "path_c": round(float(rho_c), 4), "p_c": round(float(p_c), 4),
                    "path_a": round(float(rho_a), 4), "p_a": round(float(p_a), 4),
                    "path_b": round(float(rho_b), 4), "p_b": round(float(p_b), 4),
                    "path_cp": round(float(rho_cp), 4), "p_cp": round(float(p_cp), 4),
                    "mediation": round(float(mediation), 4),
                    "prop_mediated": round(float(prop_mediated), 4),
                    "sobel_z": round(float(sobel_z), 4), "sobel_p": round(float(sobel_p), 4),
                    "verdict": verdict,
                }

    # Mediation path diagram (text-based summary)
    if results:
        print(f"\n  --- Mediation Summary ---")
        for key, r in sorted(results.items()):
            star_a = "*" if r["p_a"] < 0.05 else ""
            star_b = "*" if r["p_b"] < 0.05 else ""
            star_c = "*" if r["p_c"] < 0.05 else ""
            print(f"    {key}: c={r['path_c']:+.3f}{star_c}  a={r['path_a']:+.3f}{star_a}  "
                  f"b={r['path_b']:+.3f}{star_b}  mediation={r['mediation']:+.3f}  "
                  f"prop={r['prop_mediated']:.0%}")

    return results


# ═══════════════════════════════════════════════════════════
# LBOX-FOCUSED DEEP DIVE
# ═══════════════════════════════════════════════════════════

def lbox_deep_dive(all_sae, out_dir):
    """
    Why does LBOX show significant SAE→behavioral correlation?

    1. Per-referent_occ breakdown: which crime types drive the correlation?
    2. Layer comparison: is the correlation strongest at certain layers?
    3. Feature specificity: which SAE features are most predictive?
    """
    print(f"\n{'='*70}")
    print("LBOX DEEP DIVE: Why is H2 significant here?")
    print(f"{'='*70}")

    plt = setup_plt()
    results = {}

    lbox_entries = [e for e in all_sae if e["dataset"] == "lbox"]
    if not lbox_entries:
        print("  No LBOX SAE results found.")
        return results

    behavioral = load_behavioral_summary("lbox")
    if not behavioral:
        print("  No LBOX behavioral summary found.")
        return results

    for entry in lbox_entries:
        model = entry["model"]
        layer = entry["layer"]
        sae_results = load_sae_results(entry["results_file"])
        if not sae_results:
            continue

        # Find behavioral key
        bm_key = None
        for k in behavioral.keys():
            if k == model or model in k or k in model:
                bm_key = k
                break
        if not bm_key:
            for k in behavioral.keys():
                if k.replace("-", "") == model.replace("-", ""):
                    bm_key = k
                    break
        if not bm_key:
            continue

        pi = behavioral[bm_key].get("per_identity", {})
        bl = pi.get("(baseline)", {}).get("mean_score")
        if bl is None:
            continue

        # Build joint nFEP ↔ behavioral
        nfep_by_id = defaultdict(list)
        occ_by_id = defaultdict(set)
        for r in sae_results:
            nfep_by_id[r["combined"]].append(r["nfep"])
            occ_by_id[r["combined"]].add(r["referent_occ"])

        joint = []
        for ident, nfeps in nfep_by_id.items():
            behav = pi.get(ident)
            if not behav:
                parts = ident.split("+")
                if len(parts) == 2:
                    behav = pi.get(f"{parts[1]}+{parts[0]}")
            if not behav or behav.get("mean_score") is None:
                continue
            dev = -(behav["mean_score"] - bl)  # LBOX: flip
            joint.append({
                "identity": ident,
                "nfep": float(np.mean(nfeps)),
                "behav_dev": dev,
                "abs_dev": abs(dev),
                "occs": list(occ_by_id[ident]),
            })

        if len(joint) < 10:
            continue

        print(f"\n  [{model} L{layer}] {len(joint)} identities matched")

        # 1. Per-referent_occ breakdown
        occ_groups = defaultdict(list)
        for j in joint:
            for occ in j["occs"]:
                occ_groups[occ].append(j)

        print(f"\n    Per-crime-type correlations:")
        occ_corrs = {}
        for occ, items in sorted(occ_groups.items()):
            if len(items) < 10:
                continue
            x = [it["nfep"] for it in items]
            y = [it["abs_dev"] for it in items]
            rho, p = sp.spearmanr(x, y)
            sig = "*" if p < 0.05 else ""
            print(f"      {occ:30s} ρ={rho:+.3f} p={p:.4f}{sig} (n={len(items)})")
            occ_corrs[occ] = {"rho": round(float(rho), 4), "p": round(float(p), 4), "n": len(items)}

        # 2. Overall with bootstrap CI
        x_all = [j["nfep"] for j in joint]
        y_all = [j["abs_dev"] for j in joint]
        rho_all, p_all = sp.spearmanr(x_all, y_all)
        ci = bootstrap_ci(x_all, y_all)

        print(f"\n    Overall: ρ={rho_all:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] p={p_all:.4f}")

        # 3. Which identities drive the correlation?
        # Compute leverage: removing each identity, how much does ρ change?
        influence = []
        for i, j in enumerate(joint):
            x_loo = x_all[:i] + x_all[i+1:]
            y_loo = y_all[:i] + y_all[i+1:]
            rho_loo, _ = sp.spearmanr(x_loo, y_loo)
            influence.append({
                "identity": j["identity"],
                "nfep": j["nfep"],
                "abs_dev": j["abs_dev"],
                "rho_without": round(rho_loo, 4),
                "influence": round(rho_all - rho_loo, 4),
            })

        sorted_infl = sorted(influence, key=lambda x: abs(x["influence"]), reverse=True)
        print(f"\n    Most influential identities (leave-one-out):")
        for inf in sorted_infl[:5]:
            direction = "↑" if inf["influence"] > 0 else "↓"
            print(f"      {inf['identity']:25s} Δρ={inf['influence']:+.4f}{direction} "
                  f"nfep={inf['nfep']:.4f} |dev|={inf['abs_dev']:.4f}")

        # Plot: nFEP vs behavioral deviation colored by crime type
        fig, ax = plt.subplots(figsize=(10, 8))
        colors_map = {"Obstruction of Justice": "#1565C0", "Fraud": "#2E7D32", "Abuse": "#C62828",
                      "Suicide": "#6A1B9A", "Stress": "#E65100", "Depressed": "#00695C"}
        for j in joint:
            for occ in j["occs"]:
                c = colors_map.get(occ, "#757575")
                ax.scatter(j["nfep"], j["abs_dev"], color=c, alpha=0.5, s=40)

        # Annotate extreme points
        for j in joint:
            if j["nfep"] > np.percentile(x_all, 90) or j["abs_dev"] > np.percentile(y_all, 90):
                ax.annotate(j["identity"], (j["nfep"], j["abs_dev"]),
                           fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")

        # Regression line
        z = np.polyfit(x_all, y_all, 1)
        x_line = np.linspace(min(x_all), max(x_all), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), "--", color="gray", alpha=0.5)

        # Legend
        for occ, c in colors_map.items():
            if any(occ in j["occs"] for j in joint):
                ax.scatter([], [], color=c, label=occ, s=40)
        ax.legend(fontsize=8, loc="upper left")

        ax.set_xlabel("Delta nFEP (SAE compositional collapse)")
        ax.set_ylabel("|Behavioral Deviation| (direction-corrected)")
        ax.set_title(f"LBOX H2 Deep Dive: {model} L{layer}\n"
                     f"ρ={rho_all:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], p={p_all:.4f}")
        plt.tight_layout()
        path = out_dir / f"lbox_deep_{model}_L{layer}.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"    Saved: {path}")

        results[f"{model}_L{layer}"] = {
            "rho": round(float(rho_all), 4), "p": round(float(p_all), 4),
            "ci": ci, "n": len(joint),
            "per_occ": occ_corrs, "top_influence": sorted_infl[:10],
        }

    return results


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="LBOX-focused SAE analysis: Feature Overlap + Mediation")
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_sae = find_sae_results()
    if not all_sae:
        print("No SAE results found. Run run_sae_analysis first.")
        sys.exit(1)

    print(f"Found {len(all_sae)} SAE result sets:")
    for e in all_sae:
        print(f"  {e['dataset']}/{e['model']}/L{e['layer']}")

    # Run analyses
    overlap_results = direction1_feature_overlap(all_sae, out_dir)
    mediation_results = direction2_mediation(all_sae, out_dir)
    lbox_results = lbox_deep_dive(all_sae, out_dir)

    # Save all
    all_results = {
        "feature_overlap": overlap_results,
        "mediation": mediation_results,
        "lbox_deep_dive": lbox_results,
    }
    path = out_dir / "lbox_sae_analysis.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved: {path}")


if __name__ == "__main__":
    main()