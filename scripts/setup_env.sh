#!/usr/bin/env bash
# GPU 서버 환경 점검·정리 (한 번만 실행)
#   bash scripts/setup_env.sh
#
# 고치는 것
#   - "RuntimeError: operator torchvision::nms does not exist"
#     → torch와 torchvision 버전이 맞지 않을 때 난다. transformers가 이미지 유틸을 불러오며 torchvision을
#       import하다 실패해 Gemma2ForCausalLM / Gemma3ForConditionalGeneration import 오류로 보인다.
#       torch에 맞는 torchvision을 설치하고, 실패하면 torchvision을 제거한다(텍스트 실험에는 필요 없음).
#   - "OSError: Could not load this library: .../libtorchaudio.so" (undefined symbol)
#     → torchaudio도 같은 원인. 맞는 버전을 설치하고, 실패하면 제거한다.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "== 1) 패키지 설치 =="
pip install -q -r requirements.txt

echo "== 2a) torch CUDA 빌드 ↔ 드라이버 맞추기 =="
# pip가 최신 torch(cu130 등)를 받으면 드라이버가 낮을 때 "NVIDIA driver on your system is too old"가 난다.
# 같은 torch 버전을 드라이버가 지원하는 cu12x 빌드로 다시 설치한다 (cu12x는 12.x 드라이버에서 minor-version 호환).
DRV_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
if [ -n "$DRV_CUDA" ] && ! python -c "import torch; torch.zeros(1, device='cuda')" 2>/dev/null; then
  TORCH_BASE=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  echo "드라이버 CUDA $DRV_CUDA, torch $(python -c 'import torch; print(torch.__version__)') 가 GPU를 못 씀 → 재설치"
  for cu in cu126 cu124 cu121; do
    pip install -q --force-reinstall --no-deps "torch==$TORCH_BASE+$cu" --index-url "https://download.pytorch.org/whl/$cu" \
      && pip install -q "torch==$TORCH_BASE+$cu" --index-url "https://download.pytorch.org/whl/$cu" --extra-index-url https://pypi.org/simple \
      && python -c "import torch; torch.zeros(1, device='cuda')" 2>/dev/null && { echo "torch $TORCH_BASE+$cu 설치"; break; }
  done
  pip uninstall -y torchvision >/dev/null 2>&1   # 아래 2)에서 새 CUDA 태그로 다시 맞춘다
else
  echo "torch가 GPU 사용 가능 (드라이버 CUDA $DRV_CUDA)"
fi

echo "== 2) torch / torchvision 버전 맞추기 =="
read -r TORCH_VER CUDA_TAG < <(python - <<'EOF'
import torch
v = torch.__version__.split("+")[0]
cu = (torch.version.cuda or "").replace(".", "")
print(v, f"cu{cu}" if cu else "cpu")
EOF
)
TV_VER=$(python - "$TORCH_VER" <<'EOF'
import sys
major, minor, *rest = sys.argv[1].split(".")
patch = rest[0] if rest else "0"
print(f"0.{int(minor) + 15}.{patch}")   # torch 2.N.p ↔ torchvision 0.(N+15).p
EOF
)
echo "torch $TORCH_VER ($CUDA_TAG) → torchvision $TV_VER"
if python -c "import torchvision" 2>/dev/null; then
  echo "torchvision 정상"
else
  pip install -q "torchvision==$TV_VER" --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
    || { echo "맞는 torchvision 설치 실패 → 제거 (텍스트 실험에는 불필요)"; pip uninstall -y torchvision; }
fi

# torchaudio도 같은 문제를 낸다 (transformers.audio_utils가 import → libtorchaudio.so undefined symbol).
# torch 2.N.p ↔ torchaudio 2.N.p
echo "torch $TORCH_VER → torchaudio $TORCH_VER"
if ! pip show torchaudio >/dev/null 2>&1; then
  echo "torchaudio 없음 (정상)"
elif python -c "import torchaudio" 2>/dev/null; then
  echo "torchaudio 정상"
else
  pip install -q "torchaudio==$TORCH_VER" --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
    && python -c "import torchaudio" 2>/dev/null \
    || { echo "맞는 torchaudio 설치 실패 → 제거 (텍스트 실험에는 불필요)"; pip uninstall -y torchaudio; }
fi

echo "== 3) 모델 클래스 import 점검 =="
python - <<'EOF'
import torch, transformers
from transformers import Gemma2ForCausalLM, Gemma3ForConditionalGeneration, AutoModelForCausalLM  # noqa
print(f"transformers {transformers.__version__}, torch {torch.__version__}, GPU {torch.cuda.device_count()}개")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name} {p.total_memory // 2**30}GB")
EOF

echo "== 4) Hugging Face 로그인 (Gemma·Llama는 라이선스 동의 필요) =="
python -c "from huggingface_hub import whoami; print('로그인됨:', whoami()['name'])" 2>/dev/null \
  || echo "로그인 필요: hf auth login   (구버전: huggingface-cli login)"
