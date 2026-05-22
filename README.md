# Qwen2.5-7B LoRA SFT

LoRA fine-tuning of Qwen2.5-7B-Instruct on RTX 3090 24GB using LLaMA-Factory.

**Result**: trained ~20M params (rank=8 LoRA, target=all, 0.27% of 7.6B) on identity(91)+alpaca_zh_demo(1000), 3 epochs, eval_loss=1.3785. Model self-identity changes from "I am Qwen" to "I am shawn". See [results/before_after_demo.md](results/before_after_demo.md).

**Stack**: LLaMA-Factory 0.9.5 · PyTorch 2.8.0+cu128 · PEFT 0.18.1 · Transformers 5.6.0

**Troubleshooting**: torchaudio `libcudart.so.13` mismatch — see [docs/troubleshooting.md](docs/troubleshooting.md).

**Roadmap**: M1 done · M2 custom dataset · M3 DPO · M4 vLLM deploy
