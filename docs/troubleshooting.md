torchaudio OSError: libcudart.so.13

Cause: default torchaudio linked against CUDA 13, but PyTorch is 2.8.0+cu128.
LLaMA-Factory hard-imports torchaudio even for text-only SFT.

Fix:
  source /etc/network_turbo
  pip install torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
