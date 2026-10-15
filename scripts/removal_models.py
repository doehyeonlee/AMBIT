"""
3부(제거) 공용: 모델 · SAE 레지스트리

모델마다 다른 부분을 한곳에 모아, 같은 제거/판정 코드가 여러 모델에서 돌게 한다.
  - REMOVAL_MODELS ..... 키 → HF 경로, 공개 SAE 정보 (키 이름은 run_one_experiment와 같게 유지)
  - load_model() ....... 토크나이저 + 모델 (Gemma 3 4B 이상의 이미지-텍스트 래퍼도 처리)
  - get_layers() ....... 디코더 블록 리스트를 아키텍처와 무관하게 찾음 (model.model.layers 하드코딩 제거)
  - depth_to_layer() ... 층을 깊이 비율로 지정 → 층 수가 다른 모델에서도 같은 위치
  - format_prompt() .... run_one_experiment와 같은 시스템 프롬프트·채팅 템플릿 (Qwen3는 사고 모드 끔)
  - load_sae() ......... sae_lens / GemmaScope npz / Llama Scope safetensors / Qwen-Scope .pt 공통 인터페이스
  - sae_fidelity() ..... 베이스로 학습된 SAE를 지시 모델에 쓸 때 재구성 품질 점검

공개 SAE (2026-10 확인)
  - Gemma Scope 2: Gemma 3 270M~27B, PT·IT, 모든 층 resid_post. 25/50/65/85% 깊이는 최대 1M 폭
  - Qwen-Scope:    Qwen3-1.7B/8B, Qwen3-30B-A3B, Qwen3.5-2B/9B/27B/35B-A3B, 모든 층 (TopK). Qwen3.5-27B만 지시 모델로 학습
  - Llama Scope:   Llama-3.1-8B(Base) 모든 층
  - Gemma Scope:   Gemma-2-9B IT SAE (run_sae_analysis와 같은 google/gemma-scope-9b-it-res)
"""
from __future__ import annotations
import re
import torch

# --------------------------------------------------------------------------- registry
REMOVAL_MODELS: dict[str, dict] = {
    # 2부(SAE ΔnFEP)와 같은 모델·같은 SAE 저장소
    "gemma-2-9b": dict(
        hf="google/gemma-2-9b-it",
        sae=dict(backend="gemmascope_npz", repo="google/gemma-scope-9b-it-res", width="16k", target_l0=70,
                 trained_on="it", layers=[9, 20, 31], mid=20)),   # IT SAE는 이 3개 층에만 공개
    # Gemma 3: Gemma Scope 2의 지시 모델 SAE를 HF에서 직접 로드 (hook = model.layers.N.output)
    #   resid_post/     : 25/50/65/85% 깊이, L0 small·medium·big, 폭 16k~1M
    #   resid_post_all/ : 모든 층, L0 small·big
    "gemma-3-4b": dict(
        hf="google/gemma-3-4b-it",
        sae=dict(backend="gemmascope2", repo="google/gemma-scope-2-4b-it", width="16k", trained_on="it",
                 layers=[9, 17, 22, 29], mid=17)),               # 넓은 sweep(L0 medium) 층 = 25/50/65/85%
    "gemma-3-12b": dict(
        hf="google/gemma-3-12b-it",
        sae=dict(backend="gemmascope2", repo="google/gemma-scope-2-12b-it", width="16k", trained_on="it",
                 layers=[12, 24, 31, 41], mid=24)),
    "gemma-3-27b": dict(
        hf="google/gemma-3-27b-it",
        sae=dict(backend="gemmascope2", repo="google/gemma-scope-2-27b-it", width="16k", trained_on="it")),
    # Llama Scope: run_sae_analysis와 같은 저장소 (베이스 모델로 학습)
    "llama-3.1-8b": dict(
        hf="meta-llama/Llama-3.1-8B-Instruct",
        sae=dict(backend="llama_scope", repo="OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x",
                 file_tmpl="Llama3_1-8B-Base-L{layer}R-8x/checkpoints/final.safetensors",
                 hp_tmpl="Llama3_1-8B-Base-L{layer}R-8x/hyperparams.json", trained_on="base",
                 layers=[8, 16, 24], mid=16)),
    # Qwen-Scope (Qwen3 / Qwen3.5)
    "qwen-3-8b": dict(
        hf="Qwen/Qwen3-8B", chat_kwargs=dict(enable_thinking=False),
        sae=dict(backend="qwen_scope", repo="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50", k=50, trained_on="base",
                 layers=[8, 18, 27], mid=18)),                    # 모든 층(0~35) 공개, Base로 학습
    "qwen-3.5-9b": dict(
        hf="Qwen/Qwen3.5-9B", chat_kwargs=dict(enable_thinking=False),
        sae=dict(backend="qwen_scope", repo="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50", k=50, trained_on="base")),
    "qwen-3.5-27b": dict(
        hf="Qwen/Qwen3.5-27B", chat_kwargs=dict(enable_thinking=False),
        sae=dict(backend="qwen_scope", repo="Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50", k=50, trained_on="it")),
    # SAE 없음: PCA / LEACE / DiffMean / 아핀 / 스티어링만
    "mistral-7b": dict(hf="mistralai/Mistral-7B-Instruct-v0.3", sae=None),
    "qwen-2.5-7b": dict(hf="Qwen/Qwen2.5-7B-Instruct", sae=None),
    "r1-distill-llama-8b": dict(hf="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", sae=None),
}


# 실험 모델 세트 — 행동 측정 · SAE 분석 · 제거를 같은 세트로 돌린다
MODEL_SETS = {
    # 최신 모델 + Gemma Scope 계열 (기본)
    "core": ["gemma-2-9b", "gemma-3-4b", "gemma-3-12b", "qwen-3-8b"],
    # 비교·확장용
    "extended": ["gemma-2-9b", "gemma-3-4b", "gemma-3-12b", "qwen-3-8b", "llama-3.1-8b", "mistral-7b",
                 "gemma-3-27b", "qwen-3.5-9b", "qwen-3.5-27b"],
}


def expand_models(spec: str) -> list[str]:
    """'core' / 'extended' / 'core,llama-3.1-8b' 처럼 세트 이름과 모델 키를 섞어 쓸 수 있다."""
    out = []
    for x in spec.split(","):
        x = x.strip()
        for m in MODEL_SETS.get(x, [x] if x else []):
            if m not in out:
                out.append(m)
    return out


def sae_layers(name: str) -> list[int]:
    """2부 SAE 분석 층 (레지스트리가 유일한 기준)."""
    return list((REMOVAL_MODELS[name].get("sae") or {}).get("layers", []))


def sae_mid_layer(name: str, model=None, frac=0.5):
    """3부 SAE 소거 층: 레지스트리의 mid(2부 가운데 층과 같음), 없으면 깊이 비율."""
    sae = REMOVAL_MODELS[name].get("sae") or {}
    if "mid" in sae:
        return sae["mid"]
    return depth_to_layer(model, frac) if model is not None else None


# --------------------------------------------------------------------------- loading
def load_model(name: str, dtype=torch.bfloat16, device_map="auto"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    spec = REMOVAL_MODELS[name]
    tok = AutoTokenizer.from_pretrained(spec["hf"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=dtype, device_map=device_map)
    try:
        model = AutoModelForCausalLM.from_pretrained(spec["hf"], **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText   # Gemma 3 멀티모달 래퍼
        model = AutoModelForImageTextToText.from_pretrained(spec["hf"], **kw)
    model.eval()
    return tok, model


def get_layers(model) -> torch.nn.ModuleList:
    """디코더 블록 리스트. Llama/Qwen/Gemma2: model.model.layers,
    Gemma3 멀티모달: model.model.language_model.layers 등. 비전 타워는 제외하고 가장 긴 ModuleList."""
    best = None
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or not name.endswith("layers"):
            continue
        if any(v in name for v in ("vision", "visual", "image", "siglip", "audio", "mm_")):
            continue
        if best is None or len(mod) > len(best[1]) or ("language_model" in name and len(mod) == len(best[1])):
            best = (name, mod)
    if best is None:
        raise ValueError("디코더 층을 찾지 못했습니다. get_layers()에 경로를 추가하세요.")
    return best[1]


def n_layers(model) -> int:
    return len(get_layers(model))


def depth_to_layer(model, frac: float) -> int:
    """깊이 비율 → 층 번호. Gemma-2-9B(42층)의 18~26층 ≈ 0.43~0.62."""
    L = n_layers(model)
    return min(L - 1, max(0, round(frac * (L - 1))))


def depth_band(model, lo=0.43, hi=0.62, step=2) -> list[int]:
    return list(range(depth_to_layer(model, lo), depth_to_layer(model, hi) + 1, step))


def format_prompt(tok, name: str, prompt: str, system: str) -> str:
    """run_one_experiment.generate()와 같은 방식: system 역할을 시도하고, 안 되면(Gemma) user에 합친다."""
    kw = REMOVAL_MODELS.get(name, {}).get("chat_kwargs", {})
    if getattr(tok, "chat_template", None):
        for msgs in ([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                     [{"role": "user", "content": f"{system}\n\n{prompt}"}]):
            try:
                return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            except Exception:
                continue
    return f"{system}\n\n{prompt}\n"


# --------------------------------------------------------------------------- SAEs
class SAEHandle:
    """공통 인터페이스: encode(h)→f (…, F), decode(f)→ĥ, W_dec (F, d), dtype, device, meta"""
    def __init__(self, encode, decode, W_dec, dtype, device, meta):
        self.encode, self.decode, self.W_dec = encode, decode, W_dec
        self.dtype, self.device, self.meta = dtype, device, meta


def _jumprelu(W_enc, b_enc, W_dec, b_dec, thr):
    """GemmaScope npz: W_enc (d, F), W_dec (F, d)."""
    enc = lambda h: (lambda pre: torch.where(pre > thr, pre, torch.zeros_like(pre)))(h @ W_enc + b_enc)
    dec = lambda f: f @ W_dec + b_dec
    return enc, dec


def _topk(W_enc_dF, b_enc, W_dec_Fd, b_dec, k, relu=True, scale=1.0):
    """TopK SAE. scale: 학습 시 입력을 정규화한 배율 (Llama Scope dataset-wise norm)."""
    def enc(h):
        pre = (h * scale) @ W_enc_dF + b_enc
        v, i = pre.topk(k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, i, torch.relu(v) if relu else v)
    def dec(f):
        return (f @ W_dec_Fd + b_dec) / scale
    return enc, dec


def _sae_lens_ids(release: str):
    try:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    except ImportError:
        from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
    d = get_pretrained_saes_directory()
    return list(d[release].saes_map.keys()) if release in d else None


def _layer_of(s):
    m = re.search(r"layer_?(\d+)", s) or re.search(r"L(\d+)", s)
    return int(m.group(1)) if m else -1


def _nearest(ids, layer, prefer=()):
    have = sorted({_layer_of(s) for s in ids if _layer_of(s) >= 0})
    L = min(have, key=lambda x: abs(x - layer))
    cands = [s for s in ids if _layer_of(s) == L]
    return max(cands, key=lambda s: sum(p in s for p in prefer)), L


def load_sae(name: str, layer: int, device="cuda", dtype=torch.float32) -> SAEHandle | None:
    """layer에 가장 가까운 층의 SAE를 불러온다. 실제 쓰인 층은 meta['layer']."""
    spec = REMOVAL_MODELS[name].get("sae")
    if not spec:
        return None
    from huggingface_hub import hf_hub_download, HfApi
    T = lambda x: torch.as_tensor(x).to(device=device, dtype=dtype)
    b = spec["backend"]

    if b == "sae_lens":
        from sae_lens import SAE
        for rel in spec["releases"]:
            ids = _sae_lens_ids(rel)
            if ids:
                break
        else:
            raise KeyError(f"sae_lens에 {spec['releases']} 릴리스가 없습니다. `pip install -U sae-lens` 후 다시 시도하세요.")
        sae_id, L = _nearest(ids, layer, spec.get("prefer", ()))
        out = SAE.from_pretrained(release=rel, sae_id=sae_id, device=device)
        sae = (out[0] if isinstance(out, tuple) else out).to(dtype)
        return SAEHandle(sae.encode, sae.decode, sae.W_dec, dtype, device,
                         dict(source=f"{rel}/{sae_id}", layer=L, trained_on=spec["trained_on"]))

    if b == "gemmascope_npz":
        import numpy as np
        files = [f for f in HfApi().list_repo_files(spec["repo"])
                 if f.endswith("params.npz") and f"width_{spec['width']}" in f]
        L = min({_layer_of(f) for f in files}, key=lambda x: abs(x - layer))
        cand = [f for f in files if _layer_of(f) == L]
        f = min(cand, key=lambda s: abs(int(re.search(r"average_l0_(\d+)", s).group(1)) - spec["target_l0"]))
        p = dict(np.load(hf_hub_download(spec["repo"], f)))
        W_dec = T(p["W_dec"])
        enc, dec = _jumprelu(T(p["W_enc"]), T(p["b_enc"]), W_dec, T(p["b_dec"]), T(p["threshold"]))
        return SAEHandle(enc, dec, W_dec, dtype, device, dict(source=f"{spec['repo']}/{f}", layer=L, trained_on="it"))

    if b == "gemmascope2":
        from safetensors.torch import load_file
        files = [f for f in HfApi().list_repo_files(spec["repo"])
                 if f.endswith("params.safetensors") and f.split("/")[0] in ("resid_post", "resid_post_all")
                 and f"width_{spec['width']}_" in f]
        L = min({_layer_of(f) for f in files}, key=lambda x: abs(x - layer))
        rank = lambda f: (f.startswith("resid_post/"), "l0_medium" in f, "l0_big" in f)   # 넓은 sweep·medium 우선
        f = max([f for f in files if _layer_of(f) == L], key=rank)
        p = load_file(hf_hub_download(spec["repo"], f), device="cpu")
        W_dec = T(p["w_dec"])
        enc, dec = _jumprelu(T(p["w_enc"]), T(p["b_enc"]), W_dec, T(p["b_dec"]), T(p["threshold"]))
        return SAEHandle(enc, dec, W_dec, dtype, device, dict(source=f"{spec['repo']}/{f}", layer=L, trained_on="it"))

    if b == "llama_scope":
        import json
        from safetensors.torch import load_file
        p = load_file(hf_hub_download(spec["repo"], spec["file_tmpl"].format(layer=layer)), device="cpu")
        hp = json.load(open(hf_hub_download(spec["repo"], spec["hp_tmpl"].format(layer=layer))))
        if "encoder.weight" in p:
            W_enc, b_enc, W_dec, b_dec = p["encoder.weight"].T, p["encoder.bias"], p["decoder.weight"].T, p.get("decoder.bias")
        else:
            W_enc, b_enc, W_dec, b_dec = p["W_enc"], p["b_enc"], p["W_dec"], p.get("b_dec")
        d = W_enc.shape[0]
        b_dec = torch.zeros(d) if b_dec is None else b_dec
        # 학습 시 입력을 평균 노름이 √d가 되도록 정규화 (dataset-wise) → 같은 배율을 적용하고 복원 시 되돌림
        norm = (hp.get("dataset_average_activation_norm") or {}).get("in")
        scale = (d ** 0.5) / norm if norm else 1.0
        W_enc, b_enc, W_dec, b_dec = T(W_enc), T(b_enc), T(W_dec), T(b_dec)
        if hp.get("act_fn") == "jumprelu":
            # 공개 체크포인트는 JumpReLU(스칼라 임계값). sparsity_include_decoder_norm이면 f·‖d_i‖로 임계값 비교
            thr = hp["jump_relu_threshold"]
            dn = W_dec.norm(dim=-1) if hp.get("sparsity_include_decoder_norm") else torch.ones_like(b_enc)
            def enc(h):
                pre = (h * scale) @ W_enc + b_enc
                return pre * ((pre * dn) > thr)
            dec = lambda f: (f @ W_dec + b_dec) / scale
            act = f"jumprelu(θ={thr:.3f})"
        else:
            k = hp.get("top_k", 50)
            enc, dec = _topk(W_enc, b_enc, W_dec, b_dec, k, relu=True, scale=scale)
            act = f"topk(k={k})"
        return SAEHandle(enc, dec, W_dec / scale, dtype, device,
                         dict(source=f"{spec['repo']} L{layer} {act}", layer=layer, trained_on="base", scale=scale))

    if b == "qwen_scope":
        sd = torch.load(hf_hub_download(spec["repo"], f"layer{layer}.sae.pt"), map_location="cpu")
        W_enc = sd["W_enc"]                                   # (F, d)
        W_dec = sd["W_dec"]
        W_dec = W_dec.T if W_dec.shape[0] != W_enc.shape[0] else W_dec   # → (F, d)
        b_dec = sd.get("b_dec", torch.zeros(W_dec.shape[1]))
        W_dec = T(W_dec)
        # 모델 카드 그대로: 상위 k개 값을 ReLU 없이 사용
        enc, dec = _topk(T(W_enc).T, T(sd["b_enc"]), W_dec, T(b_dec), spec["k"], relu=False)
        return SAEHandle(enc, dec, W_dec, dtype, device,
                         dict(source=f"{spec['repo']} layer{layer}", layer=layer, trained_on=spec["trained_on"]))

    raise ValueError(b)


def sae_fidelity(sae: SAEHandle, acts) -> dict:
    """재구성 설명 분산(EV)과 평균 L0. EV < 0.7이면 그 SAE의 소거 결과는 해석하지 않는다."""
    x = torch.as_tensor(acts).to(sae.device, sae.dtype)
    f = sae.encode(x)
    xh = sae.decode(f)
    resid = ((x - xh) ** 2).sum(-1).mean()
    total = ((x - x.mean(0)) ** 2).sum(-1).mean()
    return {"explained_variance": float(1 - resid / total), "l0": float((f != 0).float().sum(-1).mean())}
