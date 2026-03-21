"""
Normalized Cross-Model Comparison (Enhanced Section E)

1. Compute z-score deviations for all (model, context, identity) combinations
2. Full tables: every identity × model × context
3. Top-8 most favored / most penalized per context (with model breakdown)
4. Male/female single-gender baseline values
5. Radar-chart-ready Excel tables: Most Lenient × Most Severe per context
6. Excel export for external graphing

Direction convention (UNIFIED):
  JobFair:  positive z = favorable (higher score = better candidate)
  LBOX:     positive z = favorable (FLIPPED: lower raw score = more lenient)
  Mind:     positive z = favorable (FLIPPED: lower raw score = less severe)

So "Most Lenient" = highest positive z across all contexts
   "Most Severe"  = most negative z across all contexts

Usage: python -m evals.normalized_analysis [--output-dir outputs/figures]
"""

import json, argparse, sys, warnings, csv
import numpy as np
from pathlib import Path
from collections import defaultdict

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

# Display config: which models to show in graphs, in order, with display names
DISPLAY_MODELS_ORDER = [
    ("claude-haiku-4.5", "Claude-Haiku-4.5"),
    ("gpt-4.1-mini", "GPT-4.1-mini"),
    ("gemma-2-9b", "Gemma-2-9B"),
    ("llama-3.1-8b", "LLama-3.1-8B"),
    ("mistral-7b", "Mistral-7B"),
]
DISPLAY_MODEL_IDS = [m[0] for m in DISPLAY_MODELS_ORDER]
DISPLAY_MODEL_NAMES = {m[0]: m[1] for m in DISPLAY_MODELS_ORDER}


def load_consolidated(dataset):
    path = ANALYSIS_DIR / dataset / f"consolidated_{dataset}.csv"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get_models(rows):
    return sorted([c.replace("score_", "") for c in rows[0].keys() if c.startswith("score_")])


def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
    return plt


# ═══════════════════════════════════════════════════════════
# STEP 1-2: Compute all z-scores and build full table
# ═══════════════════════════════════════════════════════════

def compute_all_z_scores(datasets):
    """
    Compute z-score deviations for all (model, context, identity).
    Returns:
      full_table: list of {context, model, identity, gender, attribute, z_dev, n}
      gender_baselines: {(context, model, gender): z_dev}
    """
    full_table = []
    gender_baselines = {}

    for ds, rows in datasets.items():
        models = get_models(rows)

        for model in models:
            # Collect all scores for this model
            all_scores = []
            for r in rows:
                col = f"score_{model}"
                val = r.get(col)
                if val is not None and val != "":
                    try:
                        all_scores.append(float(val))
                    except ValueError:
                        pass

            if len(all_scores) < 10:
                continue
            mu = np.mean(all_scores)
            sigma = np.std(all_scores, ddof=1)
            if sigma < 0.001:
                continue

            # Z-score every record
            z_records = []
            for r in rows:
                col = f"score_{model}"
                val = r.get(col)
                if val is None or val == "":
                    continue
                try:
                    score = float(val)
                except ValueError:
                    continue

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

                z_records.append({
                    "z": (score - mu) / sigma,
                    "gender": gender, "attribute": attribute, "condition": condition,
                })

            # Baseline z
            z_neutral = [r["z"] for r in z_records if r["condition"] == "neutral"]
            if not z_neutral:
                continue
            z_bl = np.mean(z_neutral)

            # Gender baselines
            for g in GENDERS:
                g_z = [r["z"] for r in z_records if r["condition"] == "single_gender" and r["gender"] == g]
                if g_z:
                    dev = np.mean(g_z) - z_bl
                    if ds in NEGATIVE_DOMAINS:
                        dev = -dev
                    gender_baselines[(ds, model, g)] = round(dev, 4)

            # Per-identity z-dev
            by_demo = defaultdict(list)
            for r in z_records:
                if r["condition"] == "intersectional" and r["gender"] and r["attribute"]:
                    demo = f"{r['gender']}+{r['attribute']}"
                    by_demo[demo].append(r["z"])

            for demo, z_vals in by_demo.items():
                if len(z_vals) < 5:
                    continue
                z_dev = np.mean(z_vals) - z_bl
                # Flip for negative domains
                if ds in NEGATIVE_DOMAINS:
                    z_dev = -z_dev

                parts = demo.split("+")
                gender = parts[0] if parts[0] in GENDERS else parts[1] if len(parts) > 1 and parts[1] in GENDERS else None
                attribute = parts[1] if parts[0] in GENDERS else parts[0]

                full_table.append({
                    "context": ds, "model": model,
                    "identity": demo, "gender": gender, "attribute": attribute,
                    "z_dev": round(z_dev, 4), "n": len(z_vals),
                })

    return full_table, gender_baselines


# ═══════════════════════════════════════════════════════════
# STEP 3: Top-8 most/least favored per context
# ═══════════════════════════════════════════════════════════

def find_top_bottom_identities(full_table, top_k=8):
    """
    For each context, find top-k most positive and top-k most negative identities
    based on MEAN z_dev across all models.
    Returns dict[context] = {"most_lenient": [...], "most_severe": [...]}
    """
    contexts = sorted(set(r["context"] for r in full_table))
    result = {}

    for ctx in contexts:
        ctx_rows = [r for r in full_table if r["context"] == ctx]

        # Mean z_dev per identity across models
        by_identity = defaultdict(list)
        for r in ctx_rows:
            by_identity[r["identity"]].append(r["z_dev"])

        mean_z = {ident: np.mean(vals) for ident, vals in by_identity.items() if len(vals) >= 2}
        sorted_ids = sorted(mean_z.items(), key=lambda x: x[1])

        most_severe = [ident for ident, _ in sorted_ids[:top_k]]  # most negative z
        most_lenient = [ident for ident, _ in sorted_ids[-top_k:]]  # most positive z

        # Build detail table: identity × model
        models = sorted(set(r["model"] for r in ctx_rows))

        def build_detail(identities):
            detail = []
            for ident in identities:
                row = {"identity": ident, "mean_z": round(mean_z[ident], 4)}
                for model in models:
                    vals = [r["z_dev"] for r in ctx_rows if r["identity"] == ident and r["model"] == model]
                    row[model] = round(np.mean(vals), 4) if vals else None
                detail.append(row)
            return detail

        result[ctx] = {
            "most_lenient": build_detail(most_lenient[::-1]),  # highest first
            "most_severe": build_detail(most_severe),  # most negative first
            "models": models,
        }

    return result


# ═══════════════════════════════════════════════════════════
# STEP 5-6: Radar-ready tables and Excel export
# ═══════════════════════════════════════════════════════════

def build_radar_tables(top_bottom, gender_baselines, full_table):
    """
    Build radar-chart-ready tables.

    10 axes per chart:
      - female (fixed at left, position 0 = 9 o'clock)
      - attr1 ... attr8 (top-8 identities, arranged clockwise)
      - male (fixed at right, position 5 = 3 o'clock)

    Each axis value = |z_dev| of that (gender+attribute) pair, clamped to 0.5.
    For gender-only axes (female, male): use gender baseline |z_dev|.
    For attribute axes: two values per model — one for female+attr, one for male+attr.
    But since radar has one value per axis per model, we average |female+attr| and |male+attr|.

    Wait — the user wants male and female as two of the 10 axes, not split.
    So each axis is ONE identity (combined), and the 8 attributes are the top-8
    combined identities (e.g., "female+transgender", "male+non-english-speaking").

    Let me re-read: "조합된 best identity 8개와 male, female 단독"
    → 8 best combined identities + male + female = 10 axes total.
    """
    contexts = sorted(top_bottom.keys())
    radar_tables = {}
    display_models = [m for m in DISPLAY_MODEL_IDS]

    for ctx in contexts:
        tb = top_bottom[ctx]
        all_models = tb["models"]
        models = [m for m in display_models if m in all_models]

        for direction in ["most_lenient", "most_severe"]:
            # Top-8 combined identities for this direction
            top8 = [d["identity"] for d in tb[direction]][:8]

            def _lookup(ctx, model, ident):
                for r in full_table:
                    if r["context"] == ctx and r["model"] == model:
                        if r["identity"] == ident:
                            return r["z_dev"]
                        parts = ident.split("+")
                        if len(parts) == 2 and r["identity"] == f"{parts[1]}+{parts[0]}":
                            return r["z_dev"]
                return None

            # Build 10 axes: female, attr1..attr8, male
            # Arrange so female is at left (9 o'clock) and male at right (3 o'clock)
            # In polar: left = π (index 5 of 10), right = 0 (index 0 of 10)
            # So we place: male at index 0, attrs at 1-4 (top) and 6-9 (bottom), female at 5
            # This gives male=right, female=left

            # Axis order: male, attr1, attr2, attr3, attr4, female, attr5, attr6, attr7, attr8
            top_half = top8[:4]   # indices 1-4 (clockwise from male toward female)
            bottom_half = top8[4:]  # indices 6-9 (clockwise from female toward male)

            axis_order = ["male"] + top_half + ["female"] + bottom_half

            rows = []
            for axis_name in axis_order:
                if axis_name in GENDERS:
                    # Gender baseline
                    row = {"axis": axis_name, "identity": axis_name, "type": "gender_baseline"}
                    for model in models:
                        val = gender_baselines.get((ctx, model, axis_name))
                        row[model] = min(abs(val), 0.5) if val is not None else 0
                        row[f"{model}_raw"] = val if val is not None else 0
                else:
                    # Combined identity
                    row = {"axis": axis_name, "identity": axis_name, "type": direction}
                    for model in models:
                        val = _lookup(ctx, model, axis_name)
                        row[model] = min(abs(val), 0.5) if val is not None else 0
                        row[f"{model}_raw"] = val if val is not None else 0
                rows.append(row)

            radar_tables[f"{ctx}_{direction}"] = {
                "context": ctx, "direction": direction,
                "models": models, "attributes": top8,
                "rows": rows,
            }

    return radar_tables


def export_csv_tables(full_table, top_bottom, radar_tables, gender_baselines, out_dir):
    """Export all tables as CSV for Excel import."""

    # 1. Full z-score table
    path = out_dir / "z_scores_full.csv"
    fieldnames = ["context", "model", "identity", "gender", "attribute", "z_dev", "n"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(full_table, key=lambda x: (x["context"], x["model"], x["identity"])):
            w.writerow(r)
    print(f"  Saved: {path} ({len(full_table)} rows)")

    # 2. Top/bottom per context
    for ctx, tb in top_bottom.items():
        models = tb["models"]
        for direction in ["most_lenient", "most_severe"]:
            path = out_dir / f"z_top8_{ctx}_{direction}.csv"
            fieldnames = ["identity", "mean_z"] + models
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in tb[direction]:
                    w.writerow({k: v for k, v in row.items() if k in fieldnames})
            print(f"  Saved: {path}")

    # 3. Gender baselines
    path = out_dir / "z_gender_baselines.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["context", "model", "gender", "z_dev"])
        for (ctx, model, g), val in sorted(gender_baselines.items()):
            w.writerow([ctx, model, g, val])
    print(f"  Saved: {path}")

    # 4. Radar-ready tables (one CSV per context × direction)
    for key, rt in radar_tables.items():
        path = out_dir / f"z_radar_{key}.csv"
        models = rt["models"]
        # Columns: axis, type, model1_abs, model1_raw, model2_abs, ...
        fieldnames = ["axis", "identity", "type"]
        for m in models:
            fieldnames.extend([f"{m}_abs", f"{m}_raw"])

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rt["rows"]:
                out_row = {"axis": row["axis"], "identity": row["identity"], "type": row["type"]}
                for m in models:
                    out_row[f"{m}_abs"] = row.get(m, 0)
                    out_row[f"{m}_raw"] = row.get(f"{m}_raw", 0)
                w.writerow(out_row)
        print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════
# VISUALIZATION: Radar Charts
# ═══════════════════════════════════════════════════════════

def plot_radar_charts(radar_tables, out_dir):
    """Draw radar charts: one per (context, direction).
    - 4 models (no Mistral): Claude-Haiku-4.5, GPT-4.1-mini, Gemma-2-9B, LLama-3.1-8B
    - No chart titles
    - Fixed scale: 0 to 0.5 (values > 0.5 clamped to 0.5)
    - Female axes on left half, male axes on right half
    """
    plt = setup_plt()

    SCALE_MAX = 0.5

    # 5 models (including Mistral)
    chart_models = [m for m in DISPLAY_MODEL_IDS]

    MODEL_COLORS = {
        "claude-haiku-4.5": "#1565C0",
        "gpt-4.1-mini": "#C62828",
        "gemma-2-9b": "#2E7D32",
        "llama-3.1-8b": "#6A1B9A",
        "mistral-7b": "#E65100",
    }

    for key, rt in radar_tables.items():
        ctx = rt["context"]
        direction = rt["direction"]
        rows = rt["rows"]

        if not rows or len(rows) < 3:
            continue

        axes_labels = [r["axis"] for r in rows]
        n_axes = len(axes_labels)

        angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

        for mid in chart_models:
            display_name = DISPLAY_MODEL_NAMES.get(mid, mid)
            color = MODEL_COLORS.get(mid, "#757575")

            values = [min(r.get(mid, 0), SCALE_MAX) for r in rows]
            if all(v == 0 for v in values):
                continue
            values += values[:1]

            ax.plot(angles, values, '-', linewidth=1.8, label=display_name,
                    color=color, alpha=0.75)
            ax.fill(angles, values, alpha=0.06, color=color)
        
        axes_labels = [r["axis"].replace(' + ', '+\n').replace('+', '+\n') for r in rows]

        # 2. X축 눈금 설정 (줄바꿈이 적용되도록 세팅)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axes_labels, fontsize=16, #fontweight='bold', 
                           va='center', ha='center') # 정렬 추가
        ax.tick_params(axis='x', which='major', pad=35)
        ax.set_ylim(0, SCALE_MAX)
        ax.set_yticks(np.linspace(0, SCALE_MAX, 6))  # 0, 0.1, 0.2, 0.3, 0.4, 0.5
        ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, SCALE_MAX, 6)],
                          fontsize=12, alpha=0.5)

        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.08), fontsize=9)

        plt.tight_layout()
        path = out_dir / f"radar_{key}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Normalized z-score analysis with radar charts")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--top-k", type=int, default=8, help="Number of top/bottom identities per context")
    args = p.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    datasets = {}
    for ds in ["jobfair", "lbox", "mind"]:
        rows = load_consolidated(ds)
        if rows:
            print(f"Loaded {ds}: {len(rows)} rows, models={get_models(rows)}")
            datasets[ds] = rows

    if not datasets:
        print("No data found. Run: python -m scripts.analyze_results --all")
        sys.exit(1)

    # Step 1-2: Compute z-scores
    print(f"\n{'='*70}")
    print("STEP 1-2: Computing z-score deviations")
    print(f"{'='*70}")
    full_table, gender_baselines = compute_all_z_scores(datasets)
    print(f"  Total records: {len(full_table)}")
    print(f"  Gender baselines: {len(gender_baselines)}")

    # Print summary per context × model
    for ctx in sorted(datasets.keys()):
        ctx_rows = [r for r in full_table if r["context"] == ctx]
        models = sorted(set(r["model"] for r in ctx_rows))
        print(f"\n  --- {ctx} ---")
        for model in models:
            m_rows = [r for r in ctx_rows if r["model"] == model]
            z_vals = [r["z_dev"] for r in m_rows]
            if z_vals:
                print(f"    {model:20s} n={len(m_rows)} z_range=[{min(z_vals):+.3f}, {max(z_vals):+.3f}] "
                      f"gap={max(z_vals)-min(z_vals):.3f}")

    # Step 3: Top/bottom identities
    print(f"\n{'='*70}")
    print(f"STEP 3: Top-{args.top_k} Most Lenient / Most Severe per Context")
    print(f"{'='*70}")
    top_bottom = find_top_bottom_identities(full_table, top_k=args.top_k)

    for ctx, tb in top_bottom.items():
        print(f"\n  --- {ctx} ---")
        print(f"\n    Most Lenient (positive z = favorable):")
        print(f"    {'Identity':30s} {'mean_z':8s}  " + "  ".join(f"{m:12s}" for m in tb["models"]))
        for row in tb["most_lenient"]:
            vals = "  ".join(f"{row.get(m, 0):+12.4f}" for m in tb["models"])
            print(f"    {row['identity']:30s} {row['mean_z']:+8.4f}  {vals}")

        print(f"\n    Most Severe (negative z = penalized):")
        print(f"    {'Identity':30s} {'mean_z':8s}  " + "  ".join(f"{m:12s}" for m in tb["models"]))
        for row in tb["most_severe"]:
            vals = "  ".join(f"{row.get(m, 0):+12.4f}" for m in tb["models"])
            print(f"    {row['identity']:30s} {row['mean_z']:+8.4f}  {vals}")

    # Step 4: Gender baselines
    print(f"\n{'='*70}")
    print("STEP 4: Gender Baselines (male/female only)")
    print(f"{'='*70}")
    for ctx in sorted(datasets.keys()):
        print(f"\n  --- {ctx} ---")
        models = sorted(set(m for (c, m, g) in gender_baselines if c == ctx))
        for model in models:
            m_val = gender_baselines.get((ctx, model, "male"), "—")
            f_val = gender_baselines.get((ctx, model, "female"), "—")
            print(f"    {model:20s} male={m_val:+.4f}  female={f_val:+.4f}" if isinstance(m_val, float) else
                  f"    {model:20s} male={m_val}  female={f_val}")

    # Step 5: Build radar tables
    print(f"\n{'='*70}")
    print("STEP 5: Building Radar-Chart Tables")
    print(f"{'='*70}")
    radar_tables = build_radar_tables(top_bottom, gender_baselines, full_table)

    # Step 6: Export CSVs
    print(f"\n{'='*70}")
    print("STEP 6: Exporting CSV Tables")
    print(f"{'='*70}")
    export_csv_tables(full_table, top_bottom, radar_tables, gender_baselines, out_dir)

    # Plot radar charts
    print(f"\n{'='*70}")
    print("PLOTTING: Radar Charts")
    print(f"{'='*70}")
    plot_radar_charts(radar_tables, out_dir)

    # Save JSON summary
    summary = {
        "n_records": len(full_table),
        "contexts": sorted(datasets.keys()),
        "top_bottom": {ctx: {
            "most_lenient": [d["identity"] for d in tb["most_lenient"]],
            "most_severe": [d["identity"] for d in tb["most_severe"]],
        } for ctx, tb in top_bottom.items()},
        "gender_baselines": {f"{c}_{m}_{g}": v for (c, m, g), v in gender_baselines.items()},
    }
    with open(out_dir / "normalized_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {out_dir / 'normalized_analysis_summary.json'}")


if __name__ == "__main__":
    main()