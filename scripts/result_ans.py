"""
Statistical testing and visualization for behavioral bias results.
Sections A-G: Baseline → FEP → OLS Interaction → Cross-Context → Normalization → Special → Summary

Reads: outputs/analysis/{jobfair,lbox}/consolidated_*.csv
Usage: python -m scripts.result_ans [--output-dir outputs/figures]
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

# Domains where higher score = WORSE for the subject (flip sign for cross-comparison)
# JobFair: higher = better (favorable) → no flip
# LBOX: higher = more severe sentence → flip
# Mind: higher = more severe mental issues → flip
NEGATIVE_DOMAINS = {"lbox", "mind"}

# Attribute meta-categories for [D] reversal analysis
ATTR_META = {}
for cat, traits in TRAIT_CATEGORIES.items():
    for t in traits:
        ATTR_META[t] = cat


# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_consolidated(dataset):
    path = ANALYSIS_DIR / dataset / f"consolidated_{dataset}.csv"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_score_table(rows, models):
    """
    Build structured table: list of {model, domain, gender, attribute, scores[]}.
    Also extracts neutral (baseline), single-gender, and intersectional conditions.
    """
    table = []  # each row = one record with score
    for r in rows:
        demo = r.get("demographic", "(baseline)")
        parts = demo.split("+") if demo != "(baseline)" else []

        gender = None
        attribute = None
        condition = "neutral"

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
                    table.append({
                        "model": model,
                        "gender": gender,
                        "attribute": attribute,
                        "condition": condition,
                        "score": float(val),
                        "demo": demo,
                        "referent_occ": r.get("referent_occ", ""),
                    })
                except ValueError:
                    pass
    return table


def get_models_from_rows(rows):
    if not rows:
        return []
    return sorted([c.replace("score_", "") for c in rows[0].keys() if c.startswith("score_")])


def filter_table(table, model=None, domain=None, condition=None, gender=None, attribute=None):
    out = table
    if model:
        out = [r for r in out if r["model"] == model]
    if condition:
        out = [r for r in out if r["condition"] == condition]
    if gender:
        out = [r for r in out if r["gender"] == gender]
    if attribute:
        out = [r for r in out if r["attribute"] == attribute]
    return out


def scores_of(table, **kwargs):
    return [r["score"] for r in filter_table(table, **kwargs)]


def mean_safe(arr):
    return float(np.mean(arr)) if arr else None


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return float((np.mean(group1) - np.mean(group2)) / pooled) if pooled > 0 else 0.0


def fdr_correct(p_values, alpha=0.05):
    """Apply FDR (Benjamini-Hochberg) correction. Returns (rejected, corrected_p)."""
    try:
        from statsmodels.stats.multitest import multipletests
        rejected, corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        return rejected.tolist(), corrected.tolist()
    except ImportError:
        # Manual BH procedure
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        corrected = np.zeros(n)
        rejected = [False] * n
        for rank, idx in enumerate(sorted_idx):
            corrected[idx] = p_values[idx] * n / (rank + 1)
        corrected = np.minimum.accumulate(corrected[sorted_idx[::-1]])[::-1]
        corrected = corrected[np.argsort(sorted_idx)]
        corrected = np.clip(corrected, 0, 1)
        rejected = [c < alpha for c in corrected]
        return rejected, corrected.tolist()


def bootstrap_ci(x, y, stat_func=None, n_boot=2000, ci=95):
    """Bootstrap confidence interval for a statistic on (x, y) pairs.
    Default stat_func: Spearman correlation.
    """
    if stat_func is None:
        stat_func = lambda a, b: sp.spearmanr(a, b)[0]
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


def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
    return plt


# ═══════════════════════════════════════════════════════════
# [A] BASELINE VALIDATION
# ═══════════════════════════════════════════════════════════

def section_a(tables, out_dir):
    print("\n" + "="*70)
    print("[A] BASELINE VALIDATION")
    print("="*70)

    results = {}
    for domain, table in tables.items():
        models = sorted(set(r["model"] for r in table))
        print(f"\n  --- {domain} ---")
        for model in models:
            neutral = scores_of(table, model=model, condition="neutral")
            male_only = scores_of(table, model=model, condition="single_gender", gender="male")
            female_only = scores_of(table, model=model, condition="single_gender", gender="female")

            n_mean = mean_safe(neutral)
            m_mean = mean_safe(male_only)
            f_mean = mean_safe(female_only)

            t_m, p_m = sp.ttest_ind(neutral, male_only, equal_var=False) if neutral and male_only else (0, 1)
            t_f, p_f = sp.ttest_ind(neutral, female_only, equal_var=False) if neutral and female_only else (0, 1)

            sig_m = "*" if p_m < 0.05 else ""
            sig_f = "*" if p_f < 0.05 else ""

            print(f"    {model:20s} neutral={n_mean:.3f}(n={len(neutral)})  "
                  f"male={m_mean:.3f}(p={p_m:.4f}){sig_m}  "
                  f"female={f_mean:.3f}(p={p_f:.4f}){sig_f}")

            results[f"{domain}_{model}"] = {
                "neutral_mean": n_mean, "male_mean": m_mean, "female_mean": f_mean,
                "male_p": float(p_m), "female_p": float(p_f),
                "n_neutral": len(neutral), "n_male": len(male_only), "n_female": len(female_only),
            }
    return results


# ═══════════════════════════════════════════════════════════
# [B] COMPOSITIONAL DECOMPOSITION (FEP_behavioral)
# ═══════════════════════════════════════════════════════════

def section_b(tables, out_dir):
    print("\n" + "="*70)
    print("[B] COMPOSITIONAL DECOMPOSITION (FEP_behavioral)")
    print("="*70)

    plt = setup_plt()
    all_fep = {}

    for domain, table in tables.items():
        models = sorted(set(r["model"] for r in table))
        print(f"\n  --- {domain} ---")

        for model in models:
            neutral_mean = mean_safe(scores_of(table, model=model, condition="neutral"))
            if neutral_mean is None:
                continue

            # Single-axis means
            single_g = {}
            for g in GENDERS:
                s = scores_of(table, model=model, condition="single_gender", gender=g)
                single_g[g] = mean_safe(s)

            single_a = {}
            for a in ALL_TRAITS:
                s = scores_of(table, model=model, condition="single_attribute", attribute=a)
                if s:
                    single_a[a] = mean_safe(s)

            # FEP for each intersectional identity
            fep_list = []
            for g in GENDERS:
                if single_g[g] is None:
                    continue
                for a in ALL_TRAITS:
                    if a not in single_a:
                        continue
                    observed = scores_of(table, model=model, condition="intersectional",
                                         gender=g, attribute=a)
                    if not observed:
                        continue
                    obs_mean = np.mean(observed)
                    expected = single_g[g] + single_a[a] - neutral_mean
                    fep = obs_mean - expected
                    fep_list.append({
                        "gender": g, "attribute": a,
                        "identity": f"{g}+{a}",
                        "observed": round(obs_mean, 4),
                        "expected": round(expected, 4),
                        "fep": round(float(fep), 4),
                        "n": len(observed),
                        "collapse": abs(fep) > 0.3,
                    })

            if not fep_list:
                continue

            fep_vals = [f["fep"] for f in fep_list]
            t_stat, p_val = sp.ttest_1samp(fep_vals, 0)
            n_collapse = sum(1 for f in fep_list if f["collapse"])
            n_neg = sum(1 for f in fep_list if f["fep"] < 0)

            print(f"    {model:20s} mean_FEP={np.mean(fep_vals):+.4f} "
                  f"t={t_stat:.2f} p={p_val:.4f} "
                  f"collapse={n_collapse}/{len(fep_list)} neg={n_neg}/{len(fep_list)}")

            # Top penalties
            worst = sorted(fep_list, key=lambda x: x["fep"])[:5]
            for w in worst:
                print(f"      {w['identity']:30s} FEP={w['fep']:+.4f} (obs={w['observed']:.2f} exp={w['expected']:.2f})")

            all_fep[f"{domain}_{model}"] = {
                "fep_list": fep_list, "mean_fep": float(np.mean(fep_vals)),
                "t_stat": float(t_stat), "p_value": float(p_val),
                "n_collapse": n_collapse,
            }

            # Plot
            sorted_fep = sorted(fep_list, key=lambda x: x["fep"])
            show = sorted_fep[:15] + sorted_fep[-5:]
            fig, ax = plt.subplots(figsize=(12, 8))
            labels = [x["identity"] for x in show]
            vals = [x["fep"] for x in show]
            colors = ["#d32f2f" if v < -0.1 else "#388e3c" if v > 0.1 else "#757575" for v in vals]
            ax.barh(range(len(labels)), vals, color=colors, alpha=0.8)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.axvline(-0.3, color="red", linestyle="--", alpha=0.5, label="|FEP|>0.3 collapse")
            ax.axvline(0.3, color="red", linestyle="--", alpha=0.5)
            ax.set_xlabel("FEP_behavioral")
            ax.set_title(f"Compositional Decomposition: {model} on {domain}\nmean={np.mean(fep_vals):+.4f}, p={p_val:.4f}")
            ax.legend(fontsize=8)
            plt.tight_layout()
            fig.savefig(out_dir / f"fep_behavioral_{domain}_{model}.png")
            plt.close(fig)

    print(f"  Figures saved to {out_dir}")
    return all_fep


# ═══════════════════════════════════════════════════════════
# [C] OLS INTERACTION (score ~ gender + attribute + gender:attribute)
# ═══════════════════════════════════════════════════════════

def section_c(tables, out_dir):
    print("\n" + "="*70)
    print("[C] OLS INTERACTION ANALYSIS (H4)")
    print("="*70)

    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols as sm_ols
        import pandas as pd
        HAS_SM = True
    except ImportError:
        HAS_SM = False
        print("  statsmodels not installed — falling back to manual interaction computation")

    plt = setup_plt()
    all_interactions = {}

    for domain, table in tables.items():
        models = sorted(set(r["model"] for r in table))
        print(f"\n  --- {domain} ---")

        for model in models:
            inter = [r for r in table if r["model"] == model and r["condition"] == "intersectional"
                     and r["gender"] is not None and r["attribute"] is not None]
            if len(inter) < 50:
                continue

            if HAS_SM:
                df = pd.DataFrame(inter)
                try:
                    mod = sm_ols("score ~ C(gender) + C(attribute) + C(gender):C(attribute)", data=df).fit()
                    # Extract interaction terms
                    interactions = []
                    for term, coef in mod.params.items():
                        if ":" in term:
                            p_val = mod.pvalues[term]
                            # Parse gender and attribute from term
                            interactions.append({
                                "term": term, "coef": round(float(coef), 4),
                                "p_value": float(p_val),
                                "significant": p_val < 0.05,
                            })
                    sig = sum(1 for i in interactions if i["significant"])
                    print(f"    {model:20s} {sig}/{len(interactions)} significant interactions (OLS)")
                    for i in sorted(interactions, key=lambda x: abs(x["coef"]), reverse=True)[:5]:
                        s = "*" if i["significant"] else ""
                        print(f"      {i['term']:50s} β={i['coef']:+.4f} p={i['p_value']:.4f}{s}")
                    all_interactions[f"{domain}_{model}"] = interactions
                except Exception as e:
                    print(f"    {model}: OLS failed — {e}")
            else:
                # Manual: compute interaction as FEP_behavioral per (g, a) pair
                neutral_mean = mean_safe(scores_of(table, model=model, condition="neutral"))
                if neutral_mean is None:
                    continue
                interactions = []
                for g in GENDERS:
                    g_scores = scores_of(table, model=model, condition="single_gender", gender=g)
                    g_mean = mean_safe(g_scores) or neutral_mean
                    for a in ALL_TRAITS:
                        a_scores = scores_of(table, model=model, condition="single_attribute", attribute=a)
                        a_mean = mean_safe(a_scores)
                        if a_mean is None:
                            continue
                        obs_scores = scores_of(table, model=model, condition="intersectional", gender=g, attribute=a)
                        if len(obs_scores) < 5:
                            continue
                        expected = g_mean + a_mean - neutral_mean
                        interaction = np.mean(obs_scores) - expected
                        # Test: is obs_scores mean significantly different from expected?
                        t_stat, p_val = sp.ttest_1samp(obs_scores, expected)
                        interactions.append({
                            "gender": g, "attribute": a,
                            "term": f"{g}:{a}",
                            "coef": round(float(interaction), 4),
                            "p_value": float(p_val),
                            "significant": p_val < 0.05,
                        })

                sig = sum(1 for i in interactions if i["significant"])
                print(f"    {model:20s} {sig}/{len(interactions)} significant interactions")
                for i in sorted(interactions, key=lambda x: abs(x["coef"]), reverse=True)[:5]:
                    s = "*" if i["significant"] else ""
                    print(f"      {i['term']:30s} β={i['coef']:+.4f} p={i['p_value']:.4f}{s}")
                all_interactions[f"{domain}_{model}"] = interactions

    # Cross-context interaction sign comparison
    print("\n  Cross-context interaction sign comparison:")
    models_both = set()
    for k in all_interactions:
        parts = k.split("_", 1)
        if len(parts) == 2:
            models_both.add(parts[1])
    for model in sorted(models_both):
        job_ints = {i["term"]: i["coef"] for i in all_interactions.get(f"jobfair_{model}", [])}
        lbox_ints = {i["term"]: i["coef"] for i in all_interactions.get(f"lbox_{model}", [])}
        common = set(job_ints.keys()) & set(lbox_ints.keys())
        if not common:
            continue
        reversals = sum(1 for t in common if job_ints[t] * lbox_ints[t] < 0)
        print(f"    {model}: {reversals}/{len(common)} interaction terms reverse sign between contexts")

    return all_interactions


# ═══════════════════════════════════════════════════════════
# [D] CROSS-CONTEXT REVERSAL — DEEPER ANALYSIS
# ═══════════════════════════════════════════════════════════

def section_d(tables, out_dir):
    print("\n" + "="*70)
    print("[D] CROSS-CONTEXT REVERSAL ANALYSIS (H1)")
    print("="*70)

    plt = setup_plt()

    # Build all scoring domains (exclude winoidentity)
    scoring_domains = sorted([d for d in tables.keys() if d != "winoidentity"])
    if len(scoring_domains) < 2:
        print("  Need at least 2 scoring domains.")
        return {}

    # Compute per-identity deviation for each domain × model
    def _dev(table, model, domain):
        neutral = mean_safe(scores_of(table, model=model, condition="neutral"))
        if neutral is None:
            return {}
        devs = {}
        by_demo = defaultdict(list)
        for r in table:
            if r["model"] == model and r["condition"] == "intersectional":
                demo = f"{r['gender']}+{r['attribute']}" if r["gender"] and r["attribute"] else None
                if demo:
                    by_demo[demo].append(r["score"])
        for demo, scores in by_demo.items():
            devs[demo] = np.mean(scores) - neutral
        # Flip sign for NEGATIVE_DOMAINS
        if domain in NEGATIVE_DOMAINS:
            devs = {k: -v for k, v in devs.items()}
        return devs

    results = {}

    # Compare all pairs: primary = jobfair↔lbox, also jobfair↔mind, lbox↔mind
    for di in range(len(scoring_domains)):
        for dj in range(di+1, len(scoring_domains)):
            d1, d2 = scoring_domains[di], scoring_domains[dj]
            models_both = sorted(set(r["model"] for r in tables[d1]) & set(r["model"] for r in tables[d2]))

            if not models_both:
                continue

            print(f"\n  --- {d1} ↔ {d2} ---")

            for model in models_both:
                dev1 = _dev(tables[d1], model, d1)
                dev2 = _dev(tables[d2], model, d2)
                common = sorted(set(dev1.keys()) & set(dev2.keys()))
                if not common:
                    continue

                reversals = []
                for ident in common:
                    v1, v2 = dev1[ident], dev2[ident]
                    is_reversal = (v1 * v2 < 0) if (abs(v1) > 0.01 and abs(v2) > 0.01) else False
                    mag = abs(v1 - v2)
                    parts = ident.split("+")
                    attr = parts[1] if len(parts) == 2 else parts[0]
                    meta_cat = ATTR_META.get(attr, "other")
                    if mag > 0.5:
                        strength = "strong"
                    elif mag > 0.1:
                        strength = "weak"
                    else:
                        strength = "near-tie"
                    reversals.append({
                        "identity": ident, f"{d1}_dev": round(v1, 4), f"{d2}_dev": round(v2, 4),
                        "reversal": is_reversal, "magnitude": round(mag, 4),
                        "strength": strength, "attribute": attr, "meta_category": meta_cat,
                    })

                n_rev = sum(1 for r in reversals if r["reversal"])
                n_strong = sum(1 for r in reversals if r["reversal"] and r["strength"] == "strong")
                n_weak = sum(1 for r in reversals if r["reversal"] and r["strength"] == "weak")

                # Pearson correlation + bootstrap CI
                x = [dev1[k] for k in common]
                y = [dev2[k] for k in common]
                corr, p_corr = sp.pearsonr(x, y)
                ci_lo, ci_hi = bootstrap_ci(x, y, stat_func=lambda a,b: sp.pearsonr(a,b)[0])

                print(f"\n  [{model}]")
                print(f"    Reversals: {n_rev}/{len(common)} ({n_rev/len(common)*100:.0f}%)")
                print(f"    Strong={n_strong} Weak={n_weak} Near-tie={sum(1 for r in reversals if r['strength']=='near-tie')}")
                print(f"    Pearson r={corr:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] (p={p_corr:.4f})")

                # KW by meta-category
                by_cat = defaultdict(list)
                for r in reversals:
                    if r["reversal"]:
                        by_cat[r["meta_category"]].append(r["magnitude"])
                if len(by_cat) >= 2:
                    groups = [v for v in by_cat.values() if len(v) >= 2]
                    if len(groups) >= 2:
                        H, p_kw = sp.kruskal(*groups)
                        print(f"    Reversal magnitude by category: Kruskal-Wallis H={H:.2f}, p={p_kw:.4f}")
                        for cat, mags in sorted(by_cat.items()):
                            print(f"      {cat:15s} mean_mag={np.mean(mags):.3f} (n={len(mags)})")

                # Low-reversal detail
                if n_rev < len(common) * 0.3:
                    rev_cases = [r for r in reversals if r["reversal"]]
                    print(f"    Reversal cases (minority pattern):")
                    for r in sorted(rev_cases, key=lambda x: -x["magnitude"])[:10]:
                        print(f"      {r['identity']:30s} {d1}={r[f'{d1}_dev']:+.3f} {d2}={r[f'{d2}_dev']:+.3f} "
                              f"mag={r['magnitude']:.3f} [{r['meta_category']}]")

                key = f"{d1}_{d2}_{model}"
                results[key] = {"reversals": reversals, "pearson_r": float(corr), "pearson_p": float(p_corr),
                                "pearson_ci": [ci_lo, ci_hi],
                                "n_reversals": n_rev, "n_total": len(common),
                                "domain_1": d1, "domain_2": d2}

                # Scatter plot
                fig, ax = plt.subplots(figsize=(9, 9))
                colors = ["#d32f2f" if r["reversal"] else "#1976d2" for r in reversals]
                ax.scatter([r[f"{d1}_dev"] for r in reversals], [r[f"{d2}_dev"] for r in reversals],
                          c=colors, alpha=0.6, s=30)
                for r in reversals:
                    if r["magnitude"] > 0.3:
                        ax.annotate(r["identity"], (r[f"{d1}_dev"], r[f"{d2}_dev"]), fontsize=6, alpha=0.7)
                ax.axhline(0, color="gray", linewidth=0.5)
                ax.axvline(0, color="gray", linewidth=0.5)
                all_vals = [r[f"{d1}_dev"] for r in reversals] + [r[f"{d2}_dev"] for r in reversals]
                lim = max(abs(v) for v in all_vals) * 1.2 if all_vals else 1
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
                ax.set_xlabel(f"{d1} deviation (+ = favorable)")
                ax.set_ylabel(f"{d2} deviation (+ = favorable)")
                ax.set_title(f"Cross-Context: {model} {d1}↔{d2} (r={corr:.3f}, rev={n_rev}/{len(common)})")
                plt.tight_layout()
                fig.savefig(out_dir / f"cross_context_{d1}_{d2}_{model}.png")
                plt.close(fig)

    return results


# ═══════════════════════════════════════════════════════════
# [E] MODEL NORMALIZATION & CROSS-MODEL COMPARISON
# ═══════════════════════════════════════════════════════════

def section_e(tables, out_dir):
    print("\n" + "="*70)
    print("[E] NORMALIZED CROSS-MODEL COMPARISON")
    print("="*70)

    plt = setup_plt()
    results = {}

    for domain, table in tables.items():
        models = sorted(set(r["model"] for r in table))
        print(f"\n  --- {domain} ---")

        # Z-score normalize within model
        z_table = []
        for model in models:
            m_scores = [r["score"] for r in table if r["model"] == model]
            if not m_scores:
                continue
            mu, sigma = np.mean(m_scores), np.std(m_scores, ddof=1)
            if sigma < 0.001:
                continue
            for r in table:
                if r["model"] == model:
                    z_table.append({**r, "z_score": (r["score"] - mu) / sigma})

        # Re-compute deviations on z-scores
        for model in models:
            z_neutral = [r["z_score"] for r in z_table if r["model"] == model and r["condition"] == "neutral"]
            if not z_neutral:
                continue
            z_bl = np.mean(z_neutral)

            # Per-identity z-deviation
            by_demo = defaultdict(list)
            for r in z_table:
                if r["model"] == model and r["condition"] == "intersectional":
                    demo = f"{r['gender']}+{r['attribute']}" if r["gender"] and r["attribute"] else None
                    if demo:
                        by_demo[demo].append(r["z_score"])

            z_devs = {k: np.mean(v) - z_bl for k, v in by_demo.items() if len(v) >= 5}
            if not z_devs:
                continue

            # LBOX: higher score = more severe = WORSE. Flip so positive = favorable.
            if domain in FLIP_DOMAINS:
                z_devs = {k: -v for k, v in z_devs.items()}

            max_id = max(z_devs, key=z_devs.get)
            min_id = min(z_devs, key=z_devs.get)
            gap = z_devs[max_id] - z_devs[min_id]

            print(f"    {model:20s} z-gap={gap:.3f}  "
                  f"most_favored={max_id}({z_devs[max_id]:+.3f})  "
                  f"most_penalized={min_id}({z_devs[min_id]:+.3f})")

            results[f"{domain}_{model}"] = {
                "z_gap": round(gap, 4),
                "most_favored": max_id, "most_favored_z": round(z_devs[max_id], 4),
                "most_penalized": min_id, "most_penalized_z": round(z_devs[min_id], 4),
            }

        # Heatmap: traits × models (z-scores)
        fig, ax = plt.subplots(figsize=(14, 10))
        matrix = np.zeros((len(ALL_TRAITS), len(models)))
        for mi, model in enumerate(models):
            z_neutral = [r["z_score"] for r in z_table if r["model"] == model and r["condition"] == "neutral"]
            z_bl = np.mean(z_neutral) if z_neutral else 0
            for ti, trait in enumerate(ALL_TRAITS):
                trait_z = [r["z_score"] for r in z_table if r["model"] == model and r["attribute"] == trait]
                if trait_z:
                    matrix[ti, mi] = np.mean(trait_z) - z_bl

        # LBOX: flip sign so positive = favorable
        if domain in FLIP_DOMAINS:
            matrix = -matrix

        vmax = max(abs(matrix.min()), abs(matrix.max())) or 0.5
        sign_label = "(positive = favorable)" if domain in FLIP_DOMAINS else "(z-score deviation from neutral)"
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(ALL_TRAITS)))
        ax.set_yticklabels(ALL_TRAITS, fontsize=8)
        ax.set_title(f"Normalized Trait Bias: {domain} (z-score deviation from neutral)")
        plt.colorbar(im, ax=ax, shrink=0.6, label="z-deviation")
        plt.tight_layout()
        fig.savefig(out_dir / f"normalized_heatmap_{domain}.png")
        plt.close(fig)

    return results


# ═══════════════════════════════════════════════════════════
# [F] NON-ENGLISH-SPEAKING SPECIAL ANALYSIS
# ═══════════════════════════════════════════════════════════

def section_f(tables, out_dir):
    print("\n" + "="*70)
    print("[F] NON-ENGLISH-SPEAKING SPECIAL ANALYSIS")
    print("="*70)

    target = "non-english-speaking"
    comparisons = ["american", "immigrant", "english-speaking"]

    results = {}
    for domain, table in tables.items():
        models = sorted(set(r["model"] for r in table))
        print(f"\n  --- {domain} ---")

        for model in models:
            neutral_mean = mean_safe(scores_of(table, model=model, condition="neutral"))
            if neutral_mean is None:
                continue

            # Target attribute scores (intersectional)
            target_scores = [r["score"] for r in table
                           if r["model"] == model and r["attribute"] == target and r["condition"] == "intersectional"]
            if not target_scores:
                continue

            target_dev = np.mean(target_scores) - neutral_mean

            # Compare with other attributes
            attr_devs = {}
            for a in ALL_TRAITS:
                a_scores = [r["score"] for r in table
                           if r["model"] == model and r["attribute"] == a and r["condition"] == "intersectional"]
                if a_scores:
                    attr_devs[a] = np.mean(a_scores) - neutral_mean

            # Rank
            rank = sorted(attr_devs.items(), key=lambda x: x[1])
            target_rank = next(i for i, (k, _) in enumerate(rank) if k == target) + 1

            # One-way ANOVA: is non-english-speaking significantly different from others?
            other_scores = [r["score"] for r in table
                          if r["model"] == model and r["attribute"] != target
                          and r["attribute"] is not None and r["condition"] == "intersectional"]
            if other_scores and target_scores:
                t_stat, p_val = sp.ttest_ind(target_scores, other_scores, equal_var=False)
            else:
                t_stat, p_val = 0, 1

            # Compare with nationality attributes
            comp_strs = []
            for c in comparisons:
                if c in attr_devs:
                    comp_strs.append(f"{c}={attr_devs[c]:+.3f}")

            print(f"    {model:20s} nes_dev={target_dev:+.3f} rank={target_rank}/{len(rank)} "
                  f"t={t_stat:.2f} p={p_val:.4f}  {' '.join(comp_strs)}")

            results[f"{domain}_{model}"] = {
                "target_dev": round(target_dev, 4), "rank": target_rank, "total_attrs": len(rank),
                "t_stat": round(float(t_stat), 4), "p_value": float(p_val),
            }

    return results


# ═══════════════════════════════════════════════════════════
# [G] SUMMARY TABLE FOR PAPER
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# [OCC] REFERENT OCCUPATION / CRIME CATEGORY ANALYSIS
# ═══════════════════════════════════════════════════════════

OCC_LABELS = {
    "jobfair": {"FINANCE": "Finance", "HEALTHCARE": "Healthcare", "CONSTRUCTION": "Construction"},
    "lbox": {"Obstruction of Justice": "Obstruction", "Fraud": "Fraud", "Abuse": "Abuse"},
    "mind": {"Suicide": "Suicide", "Stress": "Stress", "Depressed": "Depression"},
}

# Use NEGATIVE_DOMAINS (defined at top) for sign flip
FLIP_DOMAINS = NEGATIVE_DOMAINS

def section_occ(tables, out_dir):
    """Analyze bias patterns separately by referent_occ (job sector / crime type)."""
    print("\n" + "="*70)
    print("[OCC] REFERENT OCCUPATION / CRIME CATEGORY ANALYSIS")
    print("="*70)

    plt = setup_plt()
    results = {}

    for domain, table in tables.items():
        models = sorted(set(r["model"] for r in table))
        occs = sorted(set(r["referent_occ"] for r in table if r.get("referent_occ")))
        if not occs:
            print(f"\n  --- {domain}: no referent_occ data ---")
            continue

        labels = OCC_LABELS.get(domain, {})
        print(f"\n  --- {domain} ({len(occs)} categories: {occs}) ---")

        domain_results = {}

        for model in models:
            model_rows = [r for r in table if r["model"] == model]
            neutral_all = [r["score"] for r in model_rows if r["condition"] == "neutral"]
            bl_all = np.mean(neutral_all) if neutral_all else None

            occ_stats = {}
            for occ in occs:
                occ_rows = [r for r in model_rows if r["referent_occ"] == occ]
                neutral_occ = [r["score"] for r in occ_rows if r["condition"] == "neutral"]
                inter_occ = [r for r in occ_rows if r["condition"] == "intersectional"]

                bl_occ = np.mean(neutral_occ) if neutral_occ else bl_all
                if bl_occ is None:
                    continue

                # Per-identity deviation within this occ
                by_demo = defaultdict(list)
                for r in inter_occ:
                    demo = f"{r['gender']}+{r['attribute']}" if r["gender"] and r["attribute"] else None
                    if demo:
                        by_demo[demo].append(r["score"])

                devs = {k: np.mean(v) - bl_occ for k, v in by_demo.items() if len(v) >= 3}
                if not devs:
                    continue

                # LBOX direction correction
                if domain in FLIP_DOMAINS:
                    devs = {k: -v for k, v in devs.items()}

                max_id = max(devs, key=devs.get)
                min_id = min(devs, key=devs.get)
                disparity = devs[max_id] - devs[min_id]

                # Sig count (vs occ-specific baseline)
                sig_count = 0
                for demo_key, scores in by_demo.items():
                    if len(scores) >= 5:
                        _, p = sp.ttest_1samp(scores, bl_occ)
                        if p < 0.05:
                            sig_count += 1

                occ_label = labels.get(occ, occ)
                occ_stats[occ] = {
                    "baseline": round(bl_occ, 3),
                    "n_identities": len(devs),
                    "disparity": round(disparity, 4),
                    "most_favored": max_id,
                    "most_penalized": min_id,
                    "sig_count": sig_count,
                    "favored_dev": round(devs[max_id], 4),
                    "penalized_dev": round(devs[min_id], 4),
                }

            if occ_stats:
                print(f"\n    [{model}]")
                for occ, stats in occ_stats.items():
                    occ_label = labels.get(occ, occ)
                    dir_note = " (corrected: +favorable)" if domain in FLIP_DOMAINS else ""
                    print(f"      {occ_label:20s} bl={stats['baseline']:.3f} disp={stats['disparity']:.3f} "
                          f"sig={stats['sig_count']}/{stats['n_identities']} "
                          f"best={stats['most_favored']}({stats['favored_dev']:+.3f}) "
                          f"worst={stats['most_penalized']}({stats['penalized_dev']:+.3f}){dir_note}")

                domain_results[model] = occ_stats

        # Cross-occ comparison: does bias pattern differ by occ?
        # For each model, compute Spearman between occ-level deviation vectors
        print(f"\n    Cross-category bias consistency:")
        for model in models:
            if model not in domain_results:
                continue
            occ_list = sorted(domain_results[model].keys())
            if len(occ_list) < 2:
                continue

            # Get per-identity deviation for each occ
            occ_dev_vectors = {}
            for occ in occ_list:
                occ_rows = [r for r in table if r["model"] == model and r["referent_occ"] == occ]
                neutral_occ = [r["score"] for r in occ_rows if r["condition"] == "neutral"]
                bl_occ = np.mean(neutral_occ) if neutral_occ else 0

                by_demo = defaultdict(list)
                for r in occ_rows:
                    if r["condition"] == "intersectional" and r["gender"] and r["attribute"]:
                        by_demo[f"{r['gender']}+{r['attribute']}"].append(r["score"])

                occ_dev_vectors[occ] = {k: np.mean(v) - bl_occ for k, v in by_demo.items() if len(v) >= 3}

            # Pairwise Spearman
            for i, occ_a in enumerate(occ_list):
                for occ_b in occ_list[i+1:]:
                    common = sorted(set(occ_dev_vectors[occ_a].keys()) & set(occ_dev_vectors[occ_b].keys()))
                    if len(common) < 10:
                        continue
                    x = [occ_dev_vectors[occ_a][k] for k in common]
                    y = [occ_dev_vectors[occ_b][k] for k in common]
                    rho, p = sp.spearmanr(x, y)
                    label_a = labels.get(occ_a, occ_a)
                    label_b = labels.get(occ_b, occ_b)
                    sig = "*" if p < 0.05 else ""
                    print(f"      [{model}] {label_a} ↔ {label_b}: ρ={rho:.3f} (p={p:.4f}){sig} n={len(common)}")

        results[domain] = domain_results

    # Plot: heatmap of disparity by model × occ
    for domain, domain_results in results.items():
        if not domain_results:
            continue
        labels_d = OCC_LABELS.get(domain, {})
        models_list = sorted(domain_results.keys())
        occs = sorted(set(o for m in domain_results.values() for o in m.keys()))
        occ_labels = [labels_d.get(o, o) for o in occs]

        if len(models_list) >= 2 and len(occs) >= 2:
            fig, ax = plt.subplots(figsize=(12, max(4, len(models_list) * 0.8)))
            matrix = np.zeros((len(models_list), len(occs)))
            for mi, model in enumerate(models_list):
                for oi, occ in enumerate(occs):
                    stats = domain_results.get(model, {}).get(occ, {})
                    matrix[mi, oi] = stats.get("disparity", 0)

            im = ax.imshow(matrix, cmap="Reds", aspect="auto")
            ax.set_xticks(range(len(occs)))
            ax.set_xticklabels(occ_labels, rotation=45, ha="right")
            ax.set_yticks(range(len(models_list)))
            ax.set_yticklabels(models_list, fontsize=9)
            ax.set_title(f"{domain.upper()}: Bias Disparity by Category × Model")
            plt.colorbar(im, ax=ax, shrink=0.6, label="Disparity")

            # Add text annotations
            for mi in range(len(models_list)):
                for oi in range(len(occs)):
                    ax.text(oi, mi, f"{matrix[mi, oi]:.3f}", ha="center", va="center", fontsize=8)

            plt.tight_layout()
            path = out_dir / f"occ_disparity_{domain}.png"
            fig.savefig(path)
            plt.close(fig)
            print(f"\n    Saved: {path}")

    return results


def section_g(tables, fep_results, cross_results, norm_results, out_dir):
    print("\n" + "="*70)
    print("[G] SUMMARY TABLE FOR PAPER")
    print("="*70)

    all_models = set()
    for table in tables.values():
        all_models.update(r["model"] for r in table)
    models = sorted(all_models)

    rows = []
    for model in models:
        row = {"model": model}

        # Sig counts from baseline — with FDR correction + Cohen's d
        for domain, table in tables.items():
            neutral = scores_of(table, model=model, condition="neutral")
            bl_mean = mean_safe(neutral)
            if bl_mean is None or not neutral:
                row[f"{domain}_sig"] = "—"
                row[f"{domain}_sig_fdr"] = "—"
                row[f"{domain}_max_d"] = "—"
                continue
            p_values = []
            d_values = []
            demo_names = []
            by_demo = defaultdict(list)
            for r in table:
                if r["model"] == model and r["condition"] == "intersectional":
                    demo = f"{r['gender']}+{r['attribute']}"
                    by_demo[demo].append(r["score"])
            for demo, scores in sorted(by_demo.items()):
                if len(scores) < 5:
                    continue
                _, p = sp.ttest_1samp(scores, bl_mean)
                d = cohens_d(scores, neutral)
                # LBOX: higher score = more severe = WORSE → flip d sign
                if domain in FLIP_DOMAINS:
                    d = -d
                p_values.append(float(p))
                d_values.append(d)
                demo_names.append(demo)

            total_count = len(p_values)
            sig_uncorrected = sum(1 for p in p_values if p < 0.05)

            if total_count > 0:
                rejected, corrected_p = fdr_correct(p_values)
                sig_fdr = sum(rejected)
                max_d_idx = np.argmax([abs(d) for d in d_values])
                max_d = d_values[max_d_idx]
                max_d_id = demo_names[max_d_idx]
            else:
                sig_fdr = 0
                max_d = 0
                max_d_id = "—"

            row[f"{domain}_sig"] = f"{sig_uncorrected}/{total_count}"
            row[f"{domain}_sig_fdr"] = f"{sig_fdr}/{total_count}"
            row[f"{domain}_max_d"] = f"{max_d:+.3f}"
            row[f"{domain}_max_d_id"] = max_d_id

        # FEP
        for domain in ["jobfair", "lbox", "mind"]:
            key = f"{domain}_{model}"
            if key in fep_results:
                row[f"{domain}_fep"] = f"{fep_results[key]['mean_fep']:+.3f}"
            else:
                row[f"{domain}_fep"] = "—"

        # Cross-context (show all pairs for this model)
        cross_pairs = []
        for key, cr in (cross_results or {}).items():
            if key.endswith(f"_{model}"):
                cross_pairs.append((key, cr))
        if cross_pairs:
            # Use first pair (jobfair_lbox) as primary for the table
            _, cr = cross_pairs[0]
            row["cross_r"] = f"{cr['pearson_r']:+.3f}"
            row["reversal_rate"] = f"{cr['n_reversals']}/{cr['n_total']}"
            # Store all pairs
            for i, (key, cr2) in enumerate(cross_pairs):
                pair_label = key.replace(f"_{model}", "")
                row[f"cross_r_{pair_label}"] = f"{cr2['pearson_r']:+.3f}"
                row[f"rev_{pair_label}"] = f"{cr2['n_reversals']}/{cr2['n_total']}"
        else:
            row["cross_r"] = "—"
            row["reversal_rate"] = "—"

        # Most penalized
        for domain in ["jobfair", "lbox", "mind"]:
            key = f"{domain}_{model}"
            if key in (norm_results or {}):
                row[f"{domain}_worst"] = norm_results[key]["most_penalized"]
            else:
                row[f"{domain}_worst"] = "—"

        rows.append(row)

    # Print table
    print(f"\n  {'Model':20s} {'Job Sig':10s} {'Job FDR':10s} {'Job d':10s} "
          f"{'Lbox Sig':10s} {'Lbox FDR':10s} {'Lbox d':10s} "
          f"{'Cross r':10s} {'Reversals':12s}")
    print("  " + "-" * 110)
    for row in rows:
        print(f"  {row['model']:20s} "
              f"{row.get('jobfair_sig','—'):10s} {row.get('jobfair_sig_fdr','—'):10s} {row.get('jobfair_max_d','—'):10s} "
              f"{row.get('lbox_sig','—'):10s} {row.get('lbox_sig_fdr','—'):10s} {row.get('lbox_max_d','—'):10s} "
              f"{row.get('cross_r','—'):10s} {row.get('reversal_rate','—'):12s}")
    # Worst identity table
    print(f"\n  {'Model':20s} {'Job Worst':25s} {'Job d':10s} {'Lbox Worst':25s} {'Lbox d':10s}")
    print("  " + "-" * 95)
    for row in rows:
        print(f"  {row['model']:20s} "
              f"{row.get('jobfair_worst','—'):25s} {row.get('jobfair_max_d','—'):10s} "
              f"{row.get('lbox_worst','—'):25s} {row.get('lbox_max_d','—'):10s}")

    # Save as CSV
    csv_path = out_dir / "paper_table2.csv"
    with open(csv_path, "w", newline="") as f:
        cols = ["model", "jobfair_sig", "jobfair_sig_fdr", "jobfair_max_d",
                "lbox_sig", "lbox_sig_fdr", "lbox_max_d",
                "cross_r", "reversal_rate", "jobfair_worst", "lbox_worst",
                "jobfair_max_d_id", "lbox_max_d_id"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Table saved: {csv_path}")

    return rows


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    tables = {}
    for ds in ["jobfair", "lbox", "mind"]:
        rows = load_consolidated(ds)
        if not rows:
            continue
        models = get_models_from_rows(rows)
        print(f"Loaded {ds}: {len(rows)} rows, models={models}")
        tables[ds] = build_score_table(rows, models)
        print(f"  Parsed {len(tables[ds])} scored records")

    if not tables:
        print("No data. Run: python -m scripts.analyze_results --all")
        sys.exit(1)

    # Run sections
    a_results = section_a(tables, out_dir)
    b_results = section_b(tables, out_dir)
    c_results = section_c(tables, out_dir)
    d_results = section_d(tables, out_dir)
    e_results = section_e(tables, out_dir)
    f_results = section_f(tables, out_dir)
    occ_results = section_occ(tables, out_dir)
    g_results = section_g(tables, b_results, d_results, e_results, out_dir)

    # Save all
    all_stats = {
        "A_baseline": a_results, "B_fep": {k: {kk: vv for kk, vv in v.items() if kk != "fep_list"}
                                            for k, v in b_results.items()},
        "D_cross": {k: {kk: vv for kk, vv in v.items() if kk != "reversals"} for k, v in d_results.items()},
        "E_normalized": e_results, "F_nes": f_results, "OCC": occ_results,
    }
    with open(out_dir / "all_statistical_tests.json", "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\nAll stats saved: {out_dir / 'all_statistical_tests.json'}")


if __name__ == "__main__":
    main()