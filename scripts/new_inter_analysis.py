"""
Additivity Analysis: Active evidence that intersectional bias is additive.

Three methods to move from "absence of evidence" to "evidence of absence":
  1. TOST (Two One-Sided Tests): interaction is within ±Δ equivalence margin
  2. R² Decomposition: interaction terms add negligible explanatory power
  3. Prediction Validation: additive model predicts held-out intersectional scores well

Reads: outputs/analysis/{jobfair,lbox,mind}/consolidated_*.csv
Usage: python -m scripts.new_sae_analysis [--output-dir outputs/figures]
"""

import json, argparse, sys, warnings, csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats as sp

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

TRAIT_CATEGORIES = {
    "age": ["young", "old"],
    "disability": ["able-bodied", "disabled"],
    "body": ["thin", "fat"],
    "neurodiv": ["neurotypical", "neurodivergent"],
    "nationality": ["american", "immigrant"],
    "language": ["english-speaking", "non-english-speaking"],
    "sexuality": ["heterosexual", "gay"],
    "gender_id": ["cisgender", "transgender"],
    "ses": ["rich", "poor"],
    "race": ["white", "black", "asian", "hispanic"],
    "religion": ["christian", "muslim", "jewish"],
}
ALL_TRAITS = [t for ts in TRAIT_CATEGORIES.values() for t in ts]
GENDERS = ["male", "female"]
NEGATIVE_DOMAINS = {"lbox", "mind"}


def load_consolidated(dataset):
    path = ANALYSIS_DIR / dataset / f"consolidated_{dataset}.csv"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get_models(rows):
    return sorted([c.replace("score_", "") for c in rows[0].keys() if c.startswith("score_")])


def build_records(rows, models):
    """Build flat list of {model, gender, attribute, condition, score, referent_occ}."""
    records = []
    for r in rows:
        demo = r.get("demographic", "(baseline)")
        parts = demo.split("+") if demo != "(baseline)" else []
        gender, attribute, condition = None, None, "neutral"

        if len(parts) == 0:
            condition = "neutral"
        elif len(parts) == 1:
            if parts[0] in GENDERS:
                gender = parts[0]
                condition = "single_gender"
            elif parts[0] in ALL_TRAITS:
                attribute = parts[0]
                condition = "single_attribute"
        elif len(parts) == 2:
            condition = "intersectional"
            for p in parts:
                if p in GENDERS:
                    gender = p
                elif p in ALL_TRAITS:
                    attribute = p

        for model in models:
            col = f"score_{model}"
            val = r.get(col)
            if val is not None and val != "":
                try:
                    records.append({
                        "model": model, "gender": gender, "attribute": attribute,
                        "condition": condition, "score": float(val),
                        "referent_occ": r.get("referent_occ", ""),
                    })
                except ValueError:
                    pass
    return records


def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
    return plt


# ═══════════════════════════════════════════════════════════
# METHOD 1: TOST (Two One-Sided Tests) for Equivalence
# ═══════════════════════════════════════════════════════════

def method_tost(records, domain, delta=0.2):
    """
    TOST equivalence test for each interaction term.

    H₀: |β_interaction| ≥ Δ  (effect is meaningful)
    H₁: |β_interaction| < Δ  (effect is negligible)

    If BOTH one-sided tests reject, we conclude the interaction is
    within the equivalence margin ±Δ.

    Returns per-model summary.
    """
    print(f"\n{'='*70}")
    print(f"[METHOD 1] TOST Equivalence Testing (Δ={delta})")
    print(f"  Domain: {domain}")
    print(f"  H₀: |β_inter| ≥ {delta}  →  H₁: |β_inter| < {delta}")
    print(f"{'='*70}")

    models = sorted(set(r["model"] for r in records))
    results = {}

    for model in models:
        m_recs = [r for r in records if r["model"] == model]
        neutral = [r["score"] for r in m_recs if r["condition"] == "neutral"]
        bl = np.mean(neutral) if neutral else None
        if bl is None:
            continue

        # Get gender means
        gender_means = {}
        for g in GENDERS:
            g_scores = [r["score"] for r in m_recs if r["condition"] == "single_gender" and r["gender"] == g]
            gender_means[g] = np.mean(g_scores) if g_scores else bl

        tost_results = []
        for g in GENDERS:
            for a in ALL_TRAITS:
                inter_scores = [r["score"] for r in m_recs
                               if r["condition"] == "intersectional" and r["gender"] == g and r["attribute"] == a]
                if len(inter_scores) < 10:
                    continue

                # Expected under additivity
                a_scores = [r["score"] for r in m_recs
                           if r["condition"] == "single_attribute" and r["attribute"] == a]
                a_mean = np.mean(a_scores) if a_scores else bl
                expected = gender_means[g] + a_mean - bl

                # Observed interaction
                observed_mean = np.mean(inter_scores)
                interaction = observed_mean - expected
                se = np.std(inter_scores, ddof=1) / np.sqrt(len(inter_scores))

                if se < 1e-10:
                    continue

                # TOST: two one-sided tests
                # Test 1: H₀: β ≤ -Δ → t₁ = (β - (-Δ)) / SE
                t_lower = (interaction - (-delta)) / se
                p_lower = 1 - sp.t.cdf(t_lower, df=len(inter_scores)-1)

                # Test 2: H₀: β ≥ +Δ → t₂ = (β - Δ) / SE
                t_upper = (interaction - delta) / se
                p_upper = sp.t.cdf(t_upper, df=len(inter_scores)-1)

                # TOST p-value = max(p_lower, p_upper)
                p_tost = max(p_lower, p_upper)
                equivalent = p_tost < 0.05

                tost_results.append({
                    "gender": g, "attribute": a,
                    "interaction": round(interaction, 4),
                    "se": round(se, 4),
                    "p_tost": round(p_tost, 6),
                    "equivalent": equivalent,
                    "n": len(inter_scores),
                })

        n_equiv = sum(1 for t in tost_results if t["equivalent"])
        n_total = len(tost_results)
        pct = n_equiv / n_total * 100 if n_total else 0

        print(f"\n  [{model}] {n_equiv}/{n_total} ({pct:.0f}%) interactions within ±{delta}")

        # Show non-equivalent (failed TOST)
        non_equiv = [t for t in tost_results if not t["equivalent"]]
        if non_equiv:
            print(f"    Non-equivalent ({len(non_equiv)}):")
            for t in sorted(non_equiv, key=lambda x: abs(x["interaction"]), reverse=True)[:5]:
                print(f"      {t['gender']}+{t['attribute']:20s} β={t['interaction']:+.4f} "
                      f"SE={t['se']:.4f} p_tost={t['p_tost']:.4f}")
        else:
            print(f"    ALL interactions are within equivalence margin ±{delta}")

        # Show largest interactions (even if equivalent)
        sorted_by_size = sorted(tost_results, key=lambda x: abs(x["interaction"]), reverse=True)
        print(f"    Largest |β| (top 5):")
        for t in sorted_by_size[:5]:
            eq_str = "≡" if t["equivalent"] else "≠"
            print(f"      {t['gender']}+{t['attribute']:20s} β={t['interaction']:+.4f} p_tost={t['p_tost']:.4f} {eq_str}")

        results[f"{domain}_{model}"] = {
            "n_equivalent": n_equiv, "n_total": n_total, "pct_equivalent": round(pct, 1),
            "delta": delta, "details": tost_results,
        }

    return results


# ═══════════════════════════════════════════════════════════
# METHOD 2: R² Decomposition
# ═══════════════════════════════════════════════════════════

def method_r2(records, domain):
    """
    Compare R² of additive vs full (interaction) model.

    Additive:    Score ~ gender + attribute
    Full:        Score ~ gender * attribute (= gender + attribute + gender:attribute)

    If ΔR² ≈ 0, interaction terms add no explanatory power.
    F-test for the significance of ΔR².
    """
    print(f"\n{'='*70}")
    print(f"[METHOD 2] R² Decomposition (Additive vs Interaction)")
    print(f"  Domain: {domain}")
    print(f"{'='*70}")

    models = sorted(set(r["model"] for r in records))
    results = {}

    try:
        import statsmodels.formula.api as smf
        import pandas as pd
        has_sm = True
    except ImportError:
        has_sm = False
        print("  statsmodels not available — using manual computation")

    for model in models:
        m_recs = [r for r in records if r["model"] == model and r["condition"] == "intersectional"
                  and r["gender"] and r["attribute"]]
        if len(m_recs) < 100:
            continue

        if has_sm:
            df = pd.DataFrame(m_recs)

            try:
                # Additive model
                mod_add = smf.ols("score ~ C(gender) + C(attribute)", data=df).fit()
                r2_add = mod_add.rsquared

                # Full model with interactions
                mod_full = smf.ols("score ~ C(gender) * C(attribute)", data=df).fit()
                r2_full = mod_full.rsquared

                delta_r2 = r2_full - r2_add

                # F-test for ΔR²
                n = len(df)
                p_add = mod_add.df_model  # number of params in additive
                p_full = mod_full.df_model  # number of params in full
                df_num = p_full - p_add  # additional params (interaction terms)
                df_den = n - p_full - 1

                if df_den > 0 and delta_r2 >= 0 and (1 - r2_full) > 0:
                    F = (delta_r2 / df_num) / ((1 - r2_full) / df_den)
                    p_f = 1 - sp.f.cdf(F, df_num, df_den)
                else:
                    F, p_f = 0, 1

                # AIC/BIC comparison
                aic_add = mod_add.aic
                aic_full = mod_full.aic
                bic_add = mod_add.bic
                bic_full = mod_full.bic

                print(f"\n  [{model}]")
                print(f"    Additive:    R²={r2_add:.6f}  AIC={aic_add:.1f}  BIC={bic_add:.1f}")
                print(f"    Full:        R²={r2_full:.6f}  AIC={aic_full:.1f}  BIC={bic_full:.1f}")
                print(f"    ΔR²={delta_r2:.6f}  F={F:.3f}  p={p_f:.4f}")
                print(f"    AIC diff={aic_full-aic_add:+.1f}  BIC diff={bic_full-bic_add:+.1f}")

                if delta_r2 < 0.001:
                    verdict = "Additive model sufficient (ΔR² < 0.001)"
                elif p_f > 0.05:
                    verdict = "Interaction terms not significant"
                else:
                    verdict = f"Interaction adds {delta_r2:.4f} R² (p={p_f:.4f})"

                # BIC prefers additive?
                if bic_add < bic_full:
                    verdict += " | BIC prefers additive"
                else:
                    verdict += " | BIC prefers full"

                print(f"    → {verdict}")

                results[f"{domain}_{model}"] = {
                    "r2_additive": round(r2_add, 6), "r2_full": round(r2_full, 6),
                    "delta_r2": round(delta_r2, 6), "F": round(F, 4), "p_F": round(p_f, 4),
                    "aic_add": round(aic_add, 1), "aic_full": round(aic_full, 1),
                    "bic_add": round(bic_add, 1), "bic_full": round(bic_full, 1),
                    "verdict": verdict, "n": len(df),
                }
            except Exception as e:
                print(f"  [{model}] OLS failed: {e}")
        else:
            # Manual R² computation without statsmodels
            scores = np.array([r["score"] for r in m_recs])
            ss_tot = np.sum((scores - np.mean(scores))**2)
            if ss_tot < 1e-10:
                continue

            # Additive: predict as mean(gender) + mean(attribute) - grand_mean
            grand_mean = np.mean(scores)
            gender_means = {}
            for g in GENDERS:
                g_s = [r["score"] for r in m_recs if r["gender"] == g]
                gender_means[g] = np.mean(g_s) if g_s else grand_mean
            attr_means = {}
            for a in ALL_TRAITS:
                a_s = [r["score"] for r in m_recs if r["attribute"] == a]
                if a_s:
                    attr_means[a] = np.mean(a_s)

            pred_add = np.array([gender_means.get(r["gender"], grand_mean) +
                                attr_means.get(r["attribute"], grand_mean) - grand_mean
                                for r in m_recs])
            ss_res_add = np.sum((scores - pred_add)**2)
            r2_add = 1 - ss_res_add / ss_tot

            # Full: predict as cell mean
            cell_means = {}
            for r in m_recs:
                key = f"{r['gender']}+{r['attribute']}"
                if key not in cell_means:
                    cell_means[key] = []
                cell_means[key].append(r["score"])
            cell_means = {k: np.mean(v) for k, v in cell_means.items()}

            pred_full = np.array([cell_means.get(f"{r['gender']}+{r['attribute']}", grand_mean)
                                 for r in m_recs])
            ss_res_full = np.sum((scores - pred_full)**2)
            r2_full = 1 - ss_res_full / ss_tot

            delta_r2 = r2_full - r2_add

            print(f"\n  [{model}]")
            print(f"    Additive: R²={r2_add:.6f}")
            print(f"    Full:     R²={r2_full:.6f}")
            print(f"    ΔR²={delta_r2:.6f}")

            results[f"{domain}_{model}"] = {
                "r2_additive": round(r2_add, 6), "r2_full": round(r2_full, 6),
                "delta_r2": round(delta_r2, 6), "n": len(m_recs),
            }

    return results


# ═══════════════════════════════════════════════════════════
# METHOD 3: Prediction-Based Validation (Cross-Validation)
# ═══════════════════════════════════════════════════════════

def method_prediction(records, domain, n_splits=5):
    """
    K-fold cross-validation: train additive model on k-1 folds,
    predict intersectional scores on held-out fold.

    Compare additive model vs interaction model.
    If additive model predicts equally well, interaction terms are unnecessary.
    """
    print(f"\n{'='*70}")
    print(f"[METHOD 3] Prediction-Based Validation ({n_splits}-fold CV)")
    print(f"  Domain: {domain}")
    print(f"{'='*70}")

    models = sorted(set(r["model"] for r in records))
    results = {}

    for model in models:
        m_recs = [r for r in records if r["model"] == model and r["condition"] == "intersectional"
                  and r["gender"] and r["attribute"]]
        if len(m_recs) < 100:
            continue

        scores = np.array([r["score"] for r in m_recs])
        genders = [r["gender"] for r in m_recs]
        attrs = [r["attribute"] for r in m_recs]

        # Shuffle and split
        np.random.seed(42)
        indices = np.random.permutation(len(m_recs))
        fold_size = len(indices) // n_splits

        add_errors = []
        full_errors = []
        baseline_errors = []

        for fold in range(n_splits):
            test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])

            train = [m_recs[i] for i in train_idx]
            test = [m_recs[i] for i in test_idx]

            # Train: compute gender means, attribute means, cell means
            grand_mean = np.mean([r["score"] for r in train])

            gender_means = {}
            for g in GENDERS:
                g_s = [r["score"] for r in train if r["gender"] == g]
                gender_means[g] = np.mean(g_s) if g_s else grand_mean

            attr_means = {}
            for a in ALL_TRAITS:
                a_s = [r["score"] for r in train if r["attribute"] == a]
                if a_s:
                    attr_means[a] = np.mean(a_s)

            cell_means = defaultdict(list)
            for r in train:
                cell_means[f"{r['gender']}+{r['attribute']}"].append(r["score"])
            cell_means = {k: np.mean(v) for k, v in cell_means.items()}

            # Predict on test
            for r in test:
                true = r["score"]
                g, a = r["gender"], r["attribute"]

                # Baseline: grand mean
                baseline_errors.append((true - grand_mean) ** 2)

                # Additive prediction
                pred_add = gender_means.get(g, grand_mean) + attr_means.get(a, grand_mean) - grand_mean
                add_errors.append((true - pred_add) ** 2)

                # Full (cell mean) prediction
                pred_full = cell_means.get(f"{g}+{a}", pred_add)
                full_errors.append((true - pred_full) ** 2)

        rmse_baseline = np.sqrt(np.mean(baseline_errors))
        rmse_add = np.sqrt(np.mean(add_errors))
        rmse_full = np.sqrt(np.mean(full_errors))
        mae_add = np.mean(np.sqrt(add_errors))
        mae_full = np.mean(np.sqrt(full_errors))

        improvement = (rmse_add - rmse_full) / rmse_add * 100 if rmse_add > 0 else 0

        print(f"\n  [{model}] n={len(m_recs)}")
        print(f"    Baseline (grand mean): RMSE={rmse_baseline:.4f}")
        print(f"    Additive model:        RMSE={rmse_add:.4f}")
        print(f"    Full (cell means):     RMSE={rmse_full:.4f}")
        print(f"    Improvement from interaction: {improvement:.2f}%")

        if improvement < 1:
            verdict = "Interaction terms unnecessary (<1% improvement)"
        elif improvement < 5:
            verdict = "Minimal interaction effect (1-5% improvement)"
        else:
            verdict = f"Notable interaction effect ({improvement:.1f}% improvement)"

        # Per-identity prediction error comparison
        # Which identities does the additive model predict worst?
        identity_errors = defaultdict(lambda: {"add": [], "full": []})
        np.random.seed(42)
        indices2 = np.random.permutation(len(m_recs))
        half = len(indices2) // 2
        train2 = [m_recs[i] for i in indices2[:half]]
        test2 = [m_recs[i] for i in indices2[half:]]

        gm2 = np.mean([r["score"] for r in train2])
        gm_g2 = {}
        for g in GENDERS:
            gs = [r["score"] for r in train2 if r["gender"] == g]
            gm_g2[g] = np.mean(gs) if gs else gm2
        gm_a2 = {}
        for a in ALL_TRAITS:
            as_ = [r["score"] for r in train2 if r["attribute"] == a]
            if as_:
                gm_a2[a] = np.mean(as_)
        cm2 = defaultdict(list)
        for r in train2:
            cm2[f"{r['gender']}+{r['attribute']}"].append(r["score"])
        cm2 = {k: np.mean(v) for k, v in cm2.items()}

        for r in test2:
            key = f"{r['gender']}+{r['attribute']}"
            pred_a = gm_g2.get(r["gender"], gm2) + gm_a2.get(r["attribute"], gm2) - gm2
            pred_f = cm2.get(key, pred_a)
            identity_errors[key]["add"].append(abs(r["score"] - pred_a))
            identity_errors[key]["full"].append(abs(r["score"] - pred_f))

        worst_add = sorted(
            [(k, np.mean(v["add"]), np.mean(v["full"])) for k, v in identity_errors.items() if len(v["add"]) >= 5],
            key=lambda x: x[1], reverse=True
        )
        if worst_add:
            print(f"    Worst predicted by additive (top 5):")
            for k, mae_a, mae_f in worst_add[:5]:
                diff = mae_a - mae_f
                print(f"      {k:30s} MAE_add={mae_a:.3f}  MAE_full={mae_f:.3f}  diff={diff:+.3f}")

        print(f"    → {verdict}")

        results[f"{domain}_{model}"] = {
            "rmse_baseline": round(rmse_baseline, 4),
            "rmse_additive": round(rmse_add, 4),
            "rmse_full": round(rmse_full, 4),
            "improvement_pct": round(improvement, 2),
            "verdict": verdict, "n": len(m_recs),
        }

    return results


# ═══════════════════════════════════════════════════════════
# SUMMARY & VISUALIZATION
# ═══════════════════════════════════════════════════════════

def print_summary(tost_results, r2_results, pred_results, out_dir):
    """Print combined summary table and save."""
    print(f"\n{'='*70}")
    print(f"ADDITIVITY EVIDENCE SUMMARY")
    print(f"{'='*70}")

    all_keys = sorted(set(list(tost_results.keys()) + list(r2_results.keys()) + list(pred_results.keys())))
    summary = {}

    print(f"\n  {'Key':35s} {'TOST equiv%':12s} {'ΔR²':10s} {'Pred Δ%':10s} {'Verdict':30s}")
    print(f"  {'-'*100}")

    for key in all_keys:
        tost = tost_results.get(key, {})
        r2 = r2_results.get(key, {})
        pred = pred_results.get(key, {})

        tost_pct = f"{tost.get('pct_equivalent', '—'):.0f}%" if isinstance(tost.get('pct_equivalent'), (int, float)) else "—"
        dr2 = f"{r2.get('delta_r2', '—'):.6f}" if isinstance(r2.get('delta_r2'), float) else "—"
        pred_imp = f"{pred.get('improvement_pct', '—'):.2f}%" if isinstance(pred.get('improvement_pct'), (int, float)) else "—"

        # Combined verdict
        verdicts = []
        if tost.get("pct_equivalent", 0) >= 90:
            verdicts.append("TOST✓")
        if isinstance(r2.get("delta_r2"), float) and r2["delta_r2"] < 0.001:
            verdicts.append("R²✓")
        if isinstance(pred.get("improvement_pct"), (int, float)) and pred["improvement_pct"] < 1:
            verdicts.append("Pred✓")

        combined = " + ".join(verdicts) if verdicts else "Inconclusive"
        if len(verdicts) == 3:
            combined = "★ STRONG ADDITIVITY ★"
        elif len(verdicts) == 2:
            combined = f"Moderate additivity ({' + '.join(verdicts)})"

        print(f"  {key:35s} {tost_pct:12s} {dr2:10s} {pred_imp:10s} {combined:30s}")

        summary[key] = {
            "tost_pct_equivalent": tost.get("pct_equivalent"),
            "delta_r2": r2.get("delta_r2"),
            "prediction_improvement_pct": pred.get("improvement_pct"),
            "combined_verdict": combined,
        }

    # Save
    path = out_dir / "additivity_evidence.json"
    with open(path, "w") as f:
        json.dump({"tost": tost_results, "r2": r2_results, "prediction": pred_results,
                   "summary": summary}, f, indent=2, default=str)
    print(f"\n  Saved: {path}")

    return summary


def plot_summary(tost_results, r2_results, pred_results, out_dir):
    """Create summary visualization."""
    plt = setup_plt()

    keys = sorted(set(list(tost_results.keys()) + list(r2_results.keys()) + list(pred_results.keys())))
    if not keys:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, len(keys) * 0.5)))

    # 1. TOST equivalence %
    ax = axes[0]
    tost_vals = [tost_results.get(k, {}).get("pct_equivalent", 0) for k in keys]
    colors = ["#2e7d32" if v >= 90 else "#f9a825" if v >= 70 else "#c62828" for v in tost_vals]
    ax.barh(range(len(keys)), tost_vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k.replace("_", "\n", 1) for k in keys], fontsize=8)
    ax.set_xlabel("% interactions within ±Δ")
    ax.set_title("TOST Equivalence")
    ax.axvline(90, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlim(0, 100)

    # 2. ΔR²
    ax = axes[1]
    r2_vals = [r2_results.get(k, {}).get("delta_r2", 0) for k in keys]
    colors = ["#2e7d32" if v < 0.001 else "#f9a825" if v < 0.01 else "#c62828" for v in r2_vals]
    ax.barh(range(len(keys)), r2_vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(["" for _ in keys])
    ax.set_xlabel("ΔR² (interaction − additive)")
    ax.set_title("R² Decomposition")
    ax.axvline(0.001, color="gray", linestyle="--", alpha=0.5)

    # 3. Prediction improvement
    ax = axes[2]
    pred_vals = [pred_results.get(k, {}).get("improvement_pct", 0) for k in keys]
    colors = ["#2e7d32" if v < 1 else "#f9a825" if v < 5 else "#c62828" for v in pred_vals]
    ax.barh(range(len(keys)), pred_vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(["" for _ in keys])
    ax.set_xlabel("RMSE improvement (%)")
    ax.set_title("Prediction Validation")
    ax.axvline(1, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("Additivity Evidence: Three Methods", fontsize=14, y=1.02)
    plt.tight_layout()
    path = out_dir / "additivity_evidence.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Additivity Analysis: TOST + R² + Prediction")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--delta", type=float, default=0.2, help="TOST equivalence margin")
    args = p.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    all_tost = {}
    all_r2 = {}
    all_pred = {}

    for ds in ["jobfair", "lbox", "mind"]:
        rows = load_consolidated(ds)
        if not rows:
            continue
        models = get_models(rows)
        print(f"\nLoaded {ds}: {len(rows)} rows, models={models}")
        records = build_records(rows, models)
        scored = [r for r in records if r["condition"] == "intersectional"]
        print(f"  {len(scored)} intersectional scored records")

        # Run three methods
        tost = method_tost(records, ds, delta=args.delta)
        r2 = method_r2(records, ds)
        pred = method_prediction(records, ds)

        all_tost.update(tost)
        all_r2.update(r2)
        all_pred.update(pred)

    # Summary
    summary = print_summary(all_tost, all_r2, all_pred, out_dir)
    plot_summary(all_tost, all_r2, all_pred, out_dir)


if __name__ == "__main__":
    main()