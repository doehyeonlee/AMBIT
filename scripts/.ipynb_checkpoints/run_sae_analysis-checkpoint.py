"""
SAE-based compositional collapse analysis.

Takes a pre-augmented dataset (WinoIdentity or JobFair format),
groups rows into quadruplets (baseline, single_a, single_b, intersectional),
extracts SAE features, and computes FEP (Feature Entanglement Penalty).

Requires GPU + transformers.
- sae_lens: used for gemma-2-2b, gemma-3-4b (SAELens registry)
- huggingface_hub: used for gemma-2-9b (direct npz download, SAELens registry incomplete)

Usage:
  python -m scripts.run_sae_analysis --data-file data/jobfair.csv --model gemma-2-9b --multi-layer
  python -m scripts.run_sae_analysis --data-file data/jobfair.csv --model gemma-3-4b --layer 17 --max-groups 50
  python -m scripts.run_sae_analysis --list-models
  python -m scripts.run_sae_analysis --model gemma-2-2b --discover-layers
"""

import json, csv, argparse, sys, time, os
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sae"

# Models with SAE support
# SAE loaded via sae_lens, model loaded via HuggingFace transformers
SAE_MODELS = {
    # ── GemmaScope 1 (Gemma 2) — confirmed working with sae_lens ──
    "gemma-2-2b": {
        "hf_id": "google/gemma-2-2b-it",
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id_tmpl": "layer_{layer}/width_16k/canonical",
        "layers": [6, 12, 18, 24],
    },
    "gemma-2-9b": {
        "hf_id": "google/gemma-2-9b-it",
        "sae_release": "gemma-scope-9b-it-res",
        "sae_id_tmpl": "layer_{layer}/width_16k/average_l0_{l0}",
        # Actual available l0 values per layer (from HF repo):
        #   layer_9/width_16k:  average_l0_88 (confirmed working)
        #   layer_20/width_16k: average_l0_14, 25, 47, 91, 189
        #   layer_31/width_16k: average_l0_14, 24, 43, 76, 142
        "layer_l0": {9: 88, 20: 91, 31: 76},
        "layers": [9, 20, 31],
        "hf_sae_repo": "google/gemma-scope-9b-it-res",
        "direct_load": True,
        "layer_l0_fallbacks": {
            9:  [88, 47],
            20: [91, 47, 25],
            31: [76, 43, 142],
        },
    },
    # ── GemmaScope 2 (Gemma 3) ──
    "gemma-3-4b": {
        "hf_id": "google/gemma-3-4b-it",
        "sae_release": "gemma-scope-2-4b-it-res",
        "sae_id_tmpl": "layer_{layer}_width_16k_l0_medium",
        "layers": [17],
    },
}


# ═══════════════════════════════════════
# DATA: Group rows into quadruplets
# ═══════════════════════════════════════

def _load_csv(filepath):
    """Load CSV rows with demographic parsing."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            demo_str = row.get("demographic_identifier", "[]")
            demo = [s.strip().strip("'\"") for s in demo_str.strip("[]").split(",") if s.strip()]
            rows.append({
                "prompt": row.get("Prompt", "").strip(),
                "referent_occ": row.get("referent_occ", ""),
                "stereotype_label": row.get("stereotype_label", ""),
                "task_type": row.get("winobias_task_type", ""),
                "pronoun": row.get("pronoun", "").strip().lower(),
                "demographic": demo,
                "n_demo": len(demo),
            })
    return rows


FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}
MALE_PRONOUNS = {"he", "him", "his", "himself"}


def _detect_format(rows):
    """Detect whether this is JobFair/LBOX (gender+trait in demo) or WinoIdentity (trait in demo, gender in pronoun)."""
    # JobFair/LBOX: has n_demo=0 (baseline), n_demo=1 (male/female), n_demo=2 (gender+trait)
    n0 = sum(1 for r in rows if r["n_demo"] == 0)
    n1 = sum(1 for r in rows if r["n_demo"] == 1)
    n2 = sum(1 for r in rows if r["n_demo"] == 2)

    # WinoIdentity: mostly n_demo=1 (trait only), few/no n_demo=0 or n_demo=2
    # And has pronoun column with she/he
    has_pronoun = sum(1 for r in rows if r["pronoun"] in FEMALE_PRONOUNS | MALE_PRONOUNS) > len(rows) * 0.5

    if n2 > len(rows) * 0.3:
        return "scoring"  # JobFair/LBOX
    elif has_pronoun and n1 > len(rows) * 0.5:
        return "winoidentity"
    else:
        return "scoring"  # default


def load_and_group(filepath, max_groups=None):
    """Load data and build analysis groups. Auto-detects format."""
    rows = _load_csv(filepath)
    fmt = _detect_format(rows)
    print(f"Detected format: {fmt}")

    if fmt == "winoidentity":
        return _group_winoidentity(rows, max_groups)
    else:
        return _group_scoring(rows, max_groups)


def _group_winoidentity(rows, max_groups=None):
    """
    Group WinoIdentity data for SAE analysis.
    Each row has demographic_identifier=[trait] and pronoun=she/he.
    We derive gender from pronoun and create groups:
      - baseline: same referent_occ, no trait (if exists) or overall reference
      - single_a: gender-only (same occ, same pronoun, no trait — may not exist)
      - single_b: trait-only (same occ, different pronoun, same trait — approx)
      - inter: gender+trait (the actual row)

    Since WinoIdentity may not have explicit baseline/single rows,
    we approximate: for each (occ, gender, trait) triplet, find:
      - baseline: row with same occ but empty demographic (if exists)
      - single_gender: row with same occ, same pronoun, different trait
      - single_trait: row with same occ, different pronoun, same trait
    """
    # Index rows
    by_occ_gender_trait = defaultdict(list)  # {(occ, gender, trait): [rows]}
    baseline_by_occ = defaultdict(list)

    for r in rows:
        pronoun = r["pronoun"]
        if pronoun in FEMALE_PRONOUNS:
            gender = "female"
        elif pronoun in MALE_PRONOUNS:
            gender = "male"
        else:
            continue

        if r["n_demo"] == 0:
            baseline_by_occ[r["referent_occ"]].append(r)
        elif r["n_demo"] == 1:
            trait = r["demographic"][0]
            by_occ_gender_trait[(r["referent_occ"], gender, trait)].append(r)

    # For single_gender, find any row with same (occ, gender) but different trait
    by_occ_gender = defaultdict(list)
    for (occ, gender, trait), rlist in by_occ_gender_trait.items():
        by_occ_gender[(occ, gender)].extend(rlist)

    # For single_trait, find any row with same (occ, trait) but different gender
    by_occ_trait = defaultdict(list)
    for (occ, gender, trait), rlist in by_occ_gender_trait.items():
        by_occ_trait[(occ, trait)].extend(rlist)

    groups = []
    seen = set()

    for (occ, gender, trait), rlist in by_occ_gender_trait.items():
        key = f"{occ}_{gender}_{trait}"
        if key in seen:
            continue
        seen.add(key)

        inter_row = rlist[0]

        # Baseline: same occ, no demographic
        bases = baseline_by_occ.get(occ, [])
        if bases:
            baseline = bases[0]
        else:
            # Use a row with same occ but different gender+trait as approx baseline
            # Pick the first available with different trait
            other_rows = [r for r in by_occ_gender.get((occ, gender), [])
                         if r["demographic"][0] != trait]
            baseline = other_rows[0] if other_rows else inter_row

        # single_a = gender effect: same occ, same gender, different trait
        gender_rows = [r for r in by_occ_gender.get((occ, gender), [])
                      if r["demographic"][0] != trait]
        single_a = gender_rows[0] if gender_rows else baseline

        # single_b = trait effect: same occ, different gender, same trait
        other_gender = "female" if gender == "male" else "male"
        trait_rows = by_occ_gender_trait.get((occ, other_gender, trait), [])
        single_b = trait_rows[0] if trait_rows else baseline

        groups.append({
            "referent_occ": occ,
            "identity_a": gender,
            "identity_b": trait,
            "combined": f"{gender}+{trait}",
            "has_single_a": len(gender_rows) > 0,
            "has_single_b": len(trait_rows) > 0,
            "prompts": {
                "baseline": baseline["prompt"],
                "single_a": single_a["prompt"],
                "single_b": single_b["prompt"],
                "inter": inter_row["prompt"],
            },
        })

        if max_groups and len(groups) >= max_groups:
            break

    print(f"Loaded {len(rows)} rows → {len(groups)} WinoIdentity analysis groups")
    return groups


def _group_scoring(rows, max_groups=None):
    """
    Group JobFair/LBOX data for SAE analysis.
    Rows have n_demo=0 (baseline), 1 (gender), 2 (gender+trait).
    """

    # Separate by identity type
    baselines = [r for r in rows if r["n_demo"] == 0]
    singles = [r for r in rows if r["n_demo"] == 1]
    multis = [r for r in rows if r["n_demo"] == 2]

    print(f"Loaded {len(rows)} rows: {len(baselines)} baseline, {len(singles)} single, {len(multis)} multi")

    # Index singles and baselines by referent_occ
    single_by_occ_trait = defaultdict(dict)  # {occ: {trait: row}}
    for r in singles:
        trait = r["demographic"][0]
        occ = r["referent_occ"]
        single_by_occ_trait[occ][trait] = r

    base_by_occ = defaultdict(list)
    for r in baselines:
        base_by_occ[r["referent_occ"]].append(r)

    # Build groups: for each multi-identity row, find matching baseline + singles
    groups = []
    seen = set()
    for r in multis:
        trait_a, trait_b = r["demographic"][0], r["demographic"][1]
        occ = r["referent_occ"]
        group_key = f"{occ}_{trait_a}_{trait_b}"

        if group_key in seen:
            continue
        seen.add(group_key)

        # Find baseline
        bases = base_by_occ.get(occ, [])
        if not bases:
            continue
        baseline = bases[0]

        # Find singles: need one for trait_a AND one for trait_b
        occ_singles = single_by_occ_trait.get(occ, {})
        single_a = occ_singles.get(trait_a)
        single_b = occ_singles.get(trait_b)

        # If single for a non-gender trait doesn't exist, that's expected
        # (singles are only male/female). Use baseline as proxy for missing single.
        if not single_a:
            single_a = baseline
        if not single_b:
            single_b = baseline

        groups.append({
            "referent_occ": occ,
            "identity_a": trait_a,
            "identity_b": trait_b,
            "combined": f"{trait_a}+{trait_b}",
            "has_single_a": trait_a in occ_singles,
            "has_single_b": trait_b in occ_singles,
            "prompts": {
                "baseline": baseline["prompt"],
                "single_a": single_a["prompt"],
                "single_b": single_b["prompt"],
                "inter": r["prompt"],
            },
        })

        if max_groups and len(groups) >= max_groups:
            break

    print(f"Built {len(groups)} analysis groups")
    return groups


# ═══════════════════════════════════════
# SAE EXTRACTION
# ═══════════════════════════════════════

class SAEExtractor:
    """
    Extract SAE features from model activations.

    SAE: loaded via sae_lens (confirmed working on A100)
    Model: loaded via HuggingFace transformers + register_forward_hook
    No TransformerLens / SAETransformerBridge needed.
    """
    def __init__(self, model_key, layer, device="cuda"):
        self.cfg = SAE_MODELS[model_key]
        self.layer = layer
        self.device = device
        self.model = None
        self.tokenizer = None
        self.sae = None

    def _resolve_sae_id(self):
        """Get sae_id — use discovered valid ID if available, else build from template."""
        # Check if discover already found the exact ID for this layer
        valid_ids = self.cfg.get("_valid_ids", {})
        if self.layer in valid_ids:
            return valid_ids[self.layer]

        # Fallback to template
        tmpl = self.cfg["sae_id_tmpl"]
        if "{l0}" in tmpl:
            layer_l0 = self.cfg.get("layer_l0", {})
            l0 = layer_l0.get(self.layer, 71)
            return tmpl.format(layer=self.layer, l0=l0)
        return tmpl.format(layer=self.layer)

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        sae_id = self._resolve_sae_id()

        # 1. Load SAE
        if self.cfg.get("direct_load"):
            # Direct HuggingFace npz loading (SAELens registry incomplete for this model)
            self._load_sae_direct(sae_id)
        else:
            # SAELens loading
            self._load_sae_saelens(sae_id)

        # 2. Load model via HuggingFace transformers
        hf_id = self.cfg["hf_id"]
        print(f"Loading {hf_id} via transformers...")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=torch.bfloat16, device_map="auto")
        self.model.eval()
        print(f"  Ready: layer={self.layer}")

    def _load_sae_saelens(self, sae_id):
        """Load SAE via sae_lens registry."""
        from sae_lens import SAE
        print(f"Loading SAE via sae_lens: release={self.cfg['sae_release']}, id={sae_id}")
        for attempt in range(2):
            try:
                self.sae = SAE.from_pretrained(
                    release=self.cfg["sae_release"],
                    sae_id=sae_id,
                    device=self.device,
                )
                break
            except Exception as e:
                if "BadZipFile" in str(type(e).__name__) or "not a zip file" in str(e):
                    print(f"  Corrupted cache. Clearing and retrying...")
                    self._clear_sae_cache(sae_id)
                else:
                    raise
        self.sae.eval()
        print(f"  SAE (sae_lens): d_sae={self.sae.cfg.d_sae}")

    def _load_sae_direct(self, sae_id):
        """Load SAE directly from HuggingFace repo npz file, with l0 fallback."""
        import torch
        from huggingface_hub import hf_hub_download

        repo_id = self.cfg["hf_sae_repo"]

        # Build list of sae_ids to try: primary first, then fallbacks
        ids_to_try = [sae_id]
        fallbacks = self.cfg.get("layer_l0_fallbacks", {}).get(self.layer, [])
        tmpl = self.cfg["sae_id_tmpl"]
        for l0 in fallbacks:
            fid = tmpl.format(layer=self.layer, l0=l0)
            if fid != sae_id:
                ids_to_try.append(fid)

        params = None
        used_id = None
        for try_id in ids_to_try:
            npz_file = f"{try_id}/params.npz"
            print(f"  Trying: {repo_id}/{npz_file}...", end=" ", flush=True)
            try:
                local_path = hf_hub_download(repo_id=repo_id, filename=npz_file)
                params = dict(np.load(local_path))
                used_id = try_id
                print("OK")
                break
            except Exception as e:
                if "404" in str(e) or "EntryNotFound" in str(type(e).__name__):
                    print("not found")
                    continue
                elif "BadZipFile" in str(e) or "not a zip file" in str(e):
                    print("corrupted, retrying...")
                    self._clear_sae_cache(try_id)
                    try:
                        local_path = hf_hub_download(repo_id=repo_id, filename=npz_file,
                                                      force_download=True)
                        params = dict(np.load(local_path))
                        used_id = try_id
                        break
                    except Exception:
                        continue
                else:
                    raise

        if params is None:
            raise RuntimeError(f"Could not load SAE for layer {self.layer}. Tried: {ids_to_try}")

        # Build minimal SAE wrapper
        W_enc = torch.tensor(params["W_enc"], dtype=torch.float32, device=self.device)
        W_dec = torch.tensor(params["W_dec"], dtype=torch.float32, device=self.device)
        b_enc = torch.tensor(params["b_enc"], dtype=torch.float32, device=self.device)
        threshold = torch.tensor(params["threshold"], dtype=torch.float32, device=self.device)

        class _DirectSAE:
            def __init__(self, W_enc, W_dec, b_enc, threshold, device):
                self.W_enc = W_enc
                self.W_dec = W_dec
                self.b_enc = b_enc
                self.threshold = threshold
                self.device = device
                self.d_sae = W_enc.shape[1]
                self.d_model = W_enc.shape[0]
            def eval(self): return self
            def encode(self, x):
                x = x.to(self.device).float()
                pre_act = x @ self.W_enc + self.b_enc
                return torch.where(pre_act > self.threshold, pre_act, torch.zeros_like(pre_act))

        self.sae = _DirectSAE(W_enc, W_dec, b_enc, threshold, self.device)
        print(f"  SAE (direct): d_model={self.sae.d_model}, d_sae={self.sae.d_sae}, id={used_id}")

    def _clear_sae_cache(self, sae_id):
        """Remove corrupted cached SAE files from HuggingFace cache."""
        import glob
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        # Find and remove params.npz files matching this SAE
        pattern = os.path.join(cache_dir, "**", "params.npz")
        for f in glob.glob(pattern, recursive=True):
            if any(part in f for part in sae_id.split("/")):
                print(f"    Removing cached: {f}")
                os.remove(f)

    def extract(self, text, token_pos=-1):
        """Extract SAE features at given token position via forward hook."""
        import torch
        activations = {}

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations["hidden"] = hidden.detach()

        handle = self.model.model.layers[self.layer].register_forward_hook(hook_fn)

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                self.model(**inputs)
        except Exception as e:
            handle.remove()
            print(f"    Model forward error: {e}")
            return None

        handle.remove()

        if "hidden" not in activations:
            print(f"    Hook did not capture activations")
            return None

        try:
            # Cast bfloat16 → float32 for SAE encode
            act = activations["hidden"][0, token_pos, :].unsqueeze(0).to(self.device).float()
            features = self.sae.encode(act)[0].cpu().numpy()
            return features
        except Exception as e:
            print(f"    SAE encode error: {e}")
            return None


# ═══════════════════════════════════════
# FEP COMPUTATION
# ═══════════════════════════════════════

def _ols_fep(target, vec_a, vec_b):
    """Core OLS FEP computation on arbitrary vectors."""
    X = np.column_stack([vec_a, vec_b])
    try:
        XtX = X.T @ X + 1e-8 * np.eye(2)
        alphas = np.linalg.solve(XtX, X.T @ target)
    except np.linalg.LinAlgError:
        alphas = np.array([0.5, 0.5])

    pred = alphas[0] * vec_a + alphas[1] * vec_b
    residual = target - pred
    fep = float(np.linalg.norm(residual))

    base_norm = np.linalg.norm(vec_a) + np.linalg.norm(vec_b)
    nfep = fep / base_norm if base_norm > 0 else 0.0

    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if r2 > 0.8:
        ctype = "linear_success"
    elif np.linalg.norm(target) < 0.3 * base_norm:
        ctype = "attenuation"
    else:
        ctype = "nonlinear_collapse"

    top_residual = np.argsort(np.abs(residual))[-20:][::-1].tolist()

    return {
        "fep": fep, "nfep": float(nfep), "r_squared": float(r2),
        "alpha_a": float(alphas[0]), "alpha_b": float(alphas[1]),
        "collapse_type": ctype, "residual_top_features": top_residual,
    }


def compute_fep(f_inter, f_a, f_b, f_base):
    """
    Compute both raw and delta (baseline-subtracted) FEP.

    Raw:   f_inter ≈ α₁·f_a + α₂·f_b
           Measures overall feature composition (dominated by shared prompt content).

    Delta: Δf_inter ≈ α₁·Δf_a + α₂·Δf_b  (where Δf = f - f_base)
           Isolates pure identity-induced changes (removes shared prompt content).

    Returns dict with both sets of metrics, prefixed raw_* and delta_*.
    """
    # Raw FEP (on full feature vectors)
    raw = _ols_fep(f_inter, f_a, f_b)

    # Delta FEP (baseline-subtracted)
    df_a = f_a - f_base
    df_b = f_b - f_base
    df_inter = f_inter - f_base
    delta = _ols_fep(df_inter, df_a, df_b)

    result = {}
    # Raw metrics (prefixed)
    for k, v in raw.items():
        result[f"raw_{k}"] = v
    # Delta metrics (prefixed) — this is the primary metric
    for k, v in delta.items():
        result[f"delta_{k}"] = v
    # Convenience aliases (delta is primary)
    result["nfep"] = delta["nfep"]
    result["r_squared"] = delta["r_squared"]
    result["collapse_type"] = delta["collapse_type"]
    # Delta norms
    result["delta_norm_a"] = float(np.linalg.norm(df_a))
    result["delta_norm_b"] = float(np.linalg.norm(df_b))
    result["delta_norm_inter"] = float(np.linalg.norm(df_inter))

    return result


# ═══════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════

def run_analysis(data_file, model_key, layer, max_groups=None, output_dir=None):
    dataset_name = Path(data_file).stem
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR / f"{dataset_name}_{model_key}_L{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and group data
    groups = load_and_group(data_file, max_groups)
    if not groups:
        print("No analysis groups found.")
        return

    # Load model + SAE
    extractor = SAEExtractor(model_key, layer)
    extractor.load()

    results = []
    collapse_counts = defaultdict(int)
    t_start = time.time()

    for i, g in enumerate(groups):
        # Extract features at last token for each condition
        f_base = extractor.extract(g["prompts"]["baseline"])
        f_a = extractor.extract(g["prompts"]["single_a"])
        f_b = extractor.extract(g["prompts"]["single_b"])
        f_inter = extractor.extract(g["prompts"]["inter"])

        if any(f is None for f in [f_base, f_a, f_b, f_inter]):
            if i < 3:
                print(f"  Skip {i}: extraction failed")
            continue

        # Compute FEP (baseline-subtracted)
        fep_result = compute_fep(f_inter, f_a, f_b, f_base)
        collapse_counts[fep_result["collapse_type"]] += 1

        results.append({
            "group_idx": i,
            "referent_occ": g["referent_occ"],
            "identity_a": g["identity_a"],
            "identity_b": g["identity_b"],
            "combined": g["combined"],
            **fep_result,
            # Raw norms for diagnostics
            "norm_base": float(np.linalg.norm(f_base)),
            "norm_a": float(np.linalg.norm(f_a)),
            "norm_b": float(np.linalg.norm(f_b)),
            "norm_inter": float(np.linalg.norm(f_inter)),
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{len(groups)} ({rate:.1f}/s) collapse={dict(collapse_counts)}")

    elapsed = time.time() - t_start
    print(f"\nDone: {len(results)} results in {elapsed/60:.1f}m")
    print(f"Collapse types: {dict(collapse_counts)}")

    # Save results
    with open(out_dir / "sae_results.json", "w") as f:
        json.dump({"model": model_key, "layer": layer, "dataset": dataset_name,
                    "n_results": len(results), "collapse_counts": dict(collapse_counts),
                    "results": results}, f, indent=2)

    # Compute summary statistics
    _compute_and_save_summary(results, out_dir, model_key, layer, dataset_name)


def _compute_and_save_summary(results, out_dir, model_key, layer, dataset_name):
    """Aggregate FEP by identity and save summary."""
    if not results:
        return

    # By combined identity
    by_identity = defaultdict(list)
    for r in results:
        by_identity[r["combined"]].append(r["nfep"])

    # By single trait (both raw and delta)
    by_trait_delta = defaultdict(list)
    by_trait_raw = defaultdict(list)
    for r in results:
        by_trait_delta[r["identity_a"]].append(r["nfep"])  # delta (primary)
        by_trait_delta[r["identity_b"]].append(r["nfep"])
        by_trait_raw[r["identity_a"]].append(r["raw_nfep"])
        by_trait_raw[r["identity_b"]].append(r["raw_nfep"])

    # By identity (both)
    by_identity_delta = defaultdict(list)
    by_identity_raw = defaultdict(list)
    for r in results:
        by_identity_delta[r["combined"]].append(r["nfep"])
        by_identity_raw[r["combined"]].append(r["raw_nfep"])

    summary = {
        "model": model_key, "layer": layer, "dataset": dataset_name,
        "n_total": len(results),
        # Delta (primary)
        "delta_nfep_mean": float(np.mean([r["nfep"] for r in results])),
        "delta_nfep_std": float(np.std([r["nfep"] for r in results])),
        "delta_r2_mean": float(np.mean([r["r_squared"] for r in results])),
        "delta_collapse": dict(defaultdict(int, {r["collapse_type"]: 0 for r in results})),
        # Raw (comparison)
        "raw_nfep_mean": float(np.mean([r["raw_nfep"] for r in results])),
        "raw_nfep_std": float(np.std([r["raw_nfep"] for r in results])),
        "raw_r2_mean": float(np.mean([r["raw_r_squared"] for r in results])),
        "raw_collapse": dict(defaultdict(int, {r["raw_collapse_type"]: 0 for r in results})),
        "by_identity": {},
        "by_trait": {},
    }

    # Recount collapse
    for r in results:
        summary["delta_collapse"][r["collapse_type"]] = \
            summary["delta_collapse"].get(r["collapse_type"], 0) + 1
        summary["raw_collapse"][r["raw_collapse_type"]] = \
            summary["raw_collapse"].get(r["raw_collapse_type"], 0) + 1

    for identity in sorted(set(by_identity_delta.keys())):
        d_nfeps = by_identity_delta[identity]
        r_nfeps = by_identity_raw[identity]
        summary["by_identity"][identity] = {
            "delta_nfep": float(np.mean(d_nfeps)),
            "raw_nfep": float(np.mean(r_nfeps)),
            "n": len(d_nfeps),
        }

    for trait in sorted(set(by_trait_delta.keys())):
        d_nfeps = by_trait_delta[trait]
        r_nfeps = by_trait_raw[trait]
        summary["by_trait"][trait] = {
            "delta_nfep": float(np.mean(d_nfeps)),
            "raw_nfep": float(np.mean(r_nfeps)),
            "n": len(d_nfeps),
        }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print top findings
    print(f"\n  === Summary: {model_key} L{layer} on {dataset_name} ===")
    print(f"  Delta nFEP: {summary['delta_nfep_mean']:.4f} ± {summary['delta_nfep_std']:.4f}  R²={summary['delta_r2_mean']:.4f}")
    print(f"  Raw   nFEP: {summary['raw_nfep_mean']:.4f} ± {summary['raw_nfep_std']:.4f}  R²={summary['raw_r2_mean']:.4f}")
    print(f"  Collapse (delta): {summary['delta_collapse']}")
    print(f"  Collapse (raw):   {summary['raw_collapse']}")

    # Top 5 most collapsed identities (delta)
    sorted_ids = sorted(summary["by_identity"].items(), key=lambda x: x[1]["delta_nfep"], reverse=True)
    print(f"\n  Top 5 highest delta nFEP (most collapsed):")
    for identity, stats in sorted_ids[:5]:
        print(f"    {identity:25s} delta={stats['delta_nfep']:.4f}  raw={stats['raw_nfep']:.4f} (n={stats['n']})")

    print(f"\n  Top 5 lowest delta nFEP (most linear):")
    for identity, stats in sorted_ids[-5:]:
        print(f"    {identity:25s} delta={stats['delta_nfep']:.4f}  raw={stats['raw_nfep']:.4f} (n={stats['n']})")

    # Trait-level
    sorted_traits = sorted(summary["by_trait"].items(), key=lambda x: x[1]["delta_nfep"], reverse=True)
    print(f"\n  nFEP by trait (delta | raw):")
    for trait, stats in sorted_traits:
        print(f"    {trait:25s} delta={stats['delta_nfep']:.4f}  raw={stats['raw_nfep']:.4f}")


# ═══════════════════════════════════════
# CLI
# ═══════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="SAE Compositional Collapse Analysis")
    p.add_argument("--data-file", type=str, help="Pre-augmented CSV")
    p.add_argument("--model", type=str, help="Model key (gemma-2-2b, gemma-2-9b)")
    p.add_argument("--layer", type=int, help="Layer number")
    p.add_argument("--max-groups", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--multi-layer", action="store_true", help="Run all layers for the model")
    p.add_argument("--discover-layers", action="store_true",
                   help="Print available SAE layer IDs for this model and exit")
    args = p.parse_args()

    if args.list_models:
        print("SAE-capable models:")
        for name, cfg in SAE_MODELS.items():
            print(f"  {name:15s} layers={cfg['layers']} release={cfg['sae_release']}")
        return

    if not args.model:
        print("Error: --model required")
        sys.exit(1)
    if args.model not in SAE_MODELS:
        print(f"Model '{args.model}' not found. Use --list-models")
        sys.exit(1)

    cfg = SAE_MODELS[args.model]

    # For models with direct_load, use hardcoded layers; otherwise discover via sae_lens
    if cfg.get("direct_load"):
        valid_layers = cfg["layers"]
        valid_ids = {}
        for l in valid_layers:
            tmpl = cfg["sae_id_tmpl"]
            if "{l0}" in tmpl:
                l0 = cfg.get("layer_l0", {}).get(l, 71)
                valid_ids[l] = tmpl.format(layer=l, l0=l0)
            else:
                valid_ids[l] = tmpl.format(layer=l)
        print(f"Using hardcoded layers for {args.model} (direct HF load):")
        for l in valid_layers:
            print(f"  layer {l:3d} -> {valid_ids[l]}")
    else:
        valid_layers, valid_ids = _discover_available_layers(args.model)

    if args.discover_layers:
        return

    if not valid_layers:
        print("Error: no valid SAE layers found.")
        sys.exit(1)

    SAE_MODELS[args.model]["_valid_ids"] = valid_ids

    if not args.data_file:
        print("Error: --data-file required")
        sys.exit(1)
    if not Path(args.data_file).exists():
        print(f"Error: {args.data_file} not found")
        sys.exit(1)

    if args.multi_layer:
        for layer in valid_layers:
            print(f"\n{'#'*60}\n# Layer {layer}\n{'#'*60}")
            run_analysis(args.data_file, args.model, layer, args.max_groups, args.output_dir)
    else:
        layer = args.layer
        if layer and layer not in valid_layers:
            print(f"Warning: layer {layer} not in valid layers {valid_layers}")
            closest = min(valid_layers, key=lambda x: abs(x - layer))
            print(f"  Using closest: {closest}")
            layer = closest
        elif not layer:
            layer = valid_layers[len(valid_layers)//2]  # default to middle
            print(f"No --layer specified, using {layer} (middle of {valid_layers})")
        run_analysis(args.data_file, args.model, layer, args.max_groups, args.output_dir)


def _discover_available_layers(model_key):
    """
    Discover valid SAE IDs for this model via sae_lens.
    Returns (sorted_layer_numbers, {layer: sae_id} dict).
    Always runs before any analysis.
    """
    from sae_lens import SAE
    cfg = SAE_MODELS[model_key]
    release = cfg["sae_release"]

    fake_id = "layer_999999_width_16k_l0_medium"
    print(f"Discovering SAE IDs for release={release}...")

    try:
        SAE.from_pretrained(release=release, sae_id=fake_id, device="cpu")
        # If somehow this doesn't error, something is wrong
        return [], {}
    except ValueError as e:
        err = str(e)
        if "Valid IDs are" in err:
            import re
            all_ids = re.findall(r"'([^']*layer_\d+[^']*)'", err)

            # Prefer smallest available width: 16k > 65k > 131k > 262k
            # (some releases like 9b-it-res only have 131k)
            for target_width in ["16k", "65k", "131k", "262k"]:
                ids_filtered = [sid for sid in all_ids if target_width in sid]
                if ids_filtered:
                    print(f"  Using width={target_width} ({len(ids_filtered)} IDs)")
                    break
            else:
                ids_filtered = all_ids
                print(f"  No width filter matched, using all {len(ids_filtered)} IDs")

            if not ids_filtered:
                print(f"  No valid SAE IDs found")
                return [], {}

            # Build layer → best sae_id mapping
            # Prefer l0 in 40-100 range (medium sparsity)
            layer_candidates = defaultdict(list)
            for sid in ids_filtered:
                m = re.search(r"layer_(\d+)", sid)
                if m:
                    layer_candidates[int(m.group(1))].append(sid)

            layer_to_id = {}
            for layer_num, candidates in layer_candidates.items():
                # Try to pick l0 closest to 70 (good balance)
                best = candidates[0]
                best_dist = 999
                for sid in candidates:
                    l0_m = re.search(r"l0_(\d+)", sid) or re.search(r"average_l0_(\d+)", sid)
                    if l0_m:
                        l0_val = int(l0_m.group(1))
                        dist = abs(l0_val - 70)
                        if dist < best_dist:
                            best = sid
                            best_dist = dist
                    elif "medium" in sid:
                        best = sid
                        best_dist = 0
                    elif "canonical" in sid and best_dist > 30:
                        best = sid
                        best_dist = 30
                layer_to_id[layer_num] = best

            valid_layers = sorted(layer_to_id.keys())
            print(f"  Found {len(valid_layers)} layers: {valid_layers}")
            for l in valid_layers:
                print(f"    layer {l:3d} -> {layer_to_id[l]}")

            return valid_layers, layer_to_id
        else:
            print(f"  Unexpected error: {err[:200]}")
            return [], {}
    except Exception as e:
        print(f"  Error: {e}")
        return [], {}


if __name__ == "__main__":
    main()