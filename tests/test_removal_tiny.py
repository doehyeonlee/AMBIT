"""
3부 제거 코드 점검 (GPU·가중치 다운로드 불필요, 토크나이저만 받음)
  python -m tests.test_removal_tiny

1) 작은 무작위 모델 5종(Llama, Qwen3, Gemma 2, Gemma 3, Gemma 3 멀티모달)에서
   - 디코더 층을 찾는지
   - 배치 채점 = 한 문항씩 채점 (왼쪽 패딩·position_ids·KV 캐시 경로)
   - 개입 hook이 출력을 바꾸고, 빠져나오면 원래대로 돌아오는지
2) data/*.csv로 run_removal 전체 파이프라인(6개 기법 + 대조군 + 판정 + 프로브)이 끝까지 도는지
3) run_sae_analysis의 새 로더 경로(direct_load="removal")와 긴 프롬프트 왼쪽 자르기
4) check_sae: 실제 CSV 프롬프트로 층별 EV·L0·차원 점검이 끝까지 도는지
"""
import sys, tempfile
import numpy as np, torch
from transformers import AutoTokenizer
from scripts import removal_models as M, removal_methods as R
from scripts.run_one_experiment import SYSTEM_PROMPT

TOK_ID = "Qwen/Qwen3-8B"   # 공개 토크나이저 ('10'이 두 토큰이라 2단계 경로까지 점검됨)


def tiny_models(V):
    from transformers import (LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM,
                              Gemma2Config, Gemma2ForCausalLM, Gemma3TextConfig, Gemma3ForCausalLM)
    c = dict(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
             num_key_value_heads=2, head_dim=16, max_position_embeddings=2048)
    ms = {"llama": LlamaForCausalLM(LlamaConfig(**c)), "qwen3": Qwen3ForCausalLM(Qwen3Config(**c)),
          "gemma2": Gemma2ForCausalLM(Gemma2Config(**c, sliding_window=16)),
          "gemma3": Gemma3ForCausalLM(Gemma3TextConfig(**c, sliding_window=16))}
    try:
        from transformers import Gemma3Config, Gemma3ForConditionalGeneration
        vc = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=28, patch_size=14)
        ms["gemma3-mm"] = Gemma3ForConditionalGeneration(Gemma3Config(text_config=dict(**c, sliding_window=16),
                                                                      vision_config=vc, mm_tokens_per_image=4))
    except Exception as e:
        print("gemma3-mm 생략:", e)
    return {k: m.eval() for k, m in ms.items()}


def unit(tok):
    M.REMOVAL_MODELS["_tiny"] = dict(hf=TOK_ID, chat_kwargs=dict(enable_thinking=False), sae=None)
    prompts = ["Rate this applicant.", "A much longer prompt about a defendant who is being tried for theft. Score it.", "Hi"]
    items = [{"text": M.format_prompt(tok, "_tiny", p, SYSTEM_PROMPT)} for p in prompts]
    for name, m in tiny_models(len(tok)).items():
        b = R.score_items(m, tok, items, batch=3)
        s = np.concatenate([R.score_items(m, tok, [it], batch=1) for it in items])
        acts = R.collect_acts(m, tok, items, [1, 3], batch=2)
        with R.Intervention(m, R.random_controls(acts), 1.0):
            b2 = R.score_items(m, tok, items, 3)
        b3 = R.score_items(m, tok, items, 3)
        ok = np.abs(b - s).max() < 1e-4 and np.abs(b2 - b).max() > 0 and np.abs(b3 - b).max() == 0
        print(f"{name:10s} 층={M.n_layers(m)} 배치-단건 차이={np.abs(b - s).max():.1e} "
              f"개입 차이={np.abs(b2 - b).max():.1e} 복원 차이={np.abs(b3 - b).max():.1e} {'OK' if ok else 'FAIL'}")
        assert ok


def pipeline(tok):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    d = tempfile.mkdtemp()
    m = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                                     num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096))
    m.save_pretrained(d); tok.save_pretrained(d)
    M.REMOVAL_MODELS["_tiny"] = dict(hf=d, chat_kwargs=dict(enable_thinking=False), sae=dict(backend="fake", trained_on="base"))

    def fake_sae(name, layer, device="cpu", dtype=torch.float32):
        g = torch.Generator().manual_seed(0)
        W_enc, W_dec = torch.randn(64, 256, generator=g) / 8, torch.randn(256, 64, generator=g) / 8
        enc, dec = M._topk(W_enc, torch.zeros(256), W_dec, torch.zeros(64), 16)
        return M.SAEHandle(enc, dec, W_dec, dtype, device, dict(source="fake", layer=layer, trained_on="base"))
    M.load_sae = fake_sae
    M.load_model.__defaults__ = (torch.float32, None)
    from scripts import run_removal
    sys.argv = ["x", "--model", "_tiny", "--n-train", "2", "--n-test", "2", "--batch", "8",
                "--identities", "male,female,female+old,male+rich,female+gay",
                "--methods", "pca,leace,diffmean,affine,steer,sae", "--alphas", "1.0", "--control-alphas", "0.5,1.0",
                "--output-dir", d + "/out"]
    run_removal.main()


def sae_extractor(tok):
    from scripts import run_sae_analysis as S
    m = tiny_models(len(tok))["gemma3-mm"] if "gemma3-mm" in tiny_models(len(tok)) else tiny_models(len(tok))["qwen3"]
    def fake_sae(name, layer, device="cpu", dtype=torch.float32):
        g = torch.Generator().manual_seed(0)
        W_enc, W_dec = torch.randn(64, 256, generator=g) / 8, torch.randn(256, 64, generator=g) / 8
        enc, dec = M._topk(W_enc, torch.zeros(256), W_dec, torch.zeros(64), 16)
        return M.SAEHandle(enc, dec, W_dec, dtype, device, dict(source="fake", layer=layer, trained_on="it"))
    M.load_sae, M.load_model = fake_sae, (lambda name, **kw: (tok, m))
    S.SAE_MODELS["_tiny"] = {"hf_id": "-", "direct_load": "removal", "removal_key": "_tiny", "layers": [2]}
    ex = S.SAEExtractor("_tiny", 2, device="cpu")
    ex.load()
    long_text = "word " * 5000 + "The higher the score, the more severe."
    f = ex.extract(long_text)
    ids = ex.tokenizer(long_text, truncation=True, max_length=4096)["input_ids"]
    keeps_tail = tok.decode(ids[-5:]).strip().endswith("severe.")
    print(f"run_sae_analysis 새 로더: 특징 {f.shape}, 활성 {int((f != 0).sum())}개, 긴 프롬프트 끝 보존={keeps_tail}")
    assert f.shape == (256,) and keeps_tail


def check(tok):
    from scripts import check_sae as C
    m = tiny_models(len(tok))["qwen3"]
    def fake_sae(name, layer, device="cpu", dtype=torch.float32):
        g = torch.Generator().manual_seed(layer)
        W_enc, W_dec = torch.randn(64, 256, generator=g) / 8, torch.randn(256, 64, generator=g) / 8
        enc, dec = M._topk(W_enc, torch.zeros(256), W_dec, torch.zeros(64), 16)
        return M.SAEHandle(enc, dec, W_dec, dtype, device, dict(source="fake", layer=layer, trained_on="base"))
    M.load_sae, M.load_model = fake_sae, (lambda name, **kw: (tok, m))
    M.REMOVAL_MODELS["_tiny"] = dict(hf="-", chat_kwargs=dict(enable_thinking=False),
                                     sae=dict(backend="fake", trained_on="base", layers=[1, 3], mid=2))
    d = tempfile.mkdtemp()
    sys.argv = ["x", "--model", "_tiny", "--n-per-ctx", "2", "--output-dir", d]
    C.main()
    rep = __import__("json").load(open(f"{d}/_tiny.json"))
    assert set(rep["layers"]) == {"1", "2", "3"} and all("raw" in r for r in rep["layers"].values())


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(TOK_ID)
    unit(tok)
    sae_extractor(tok)
    check(tok)
    pipeline(tok)
    print("\n모든 점검 통과")
