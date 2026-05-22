import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/root/autodl-tmp/saves/qwen25-7b-lora-sft"
QUESTIONS = ["你是谁？", "你叫什么名字？", "Who are you?", "你由谁开发？"]

def gen(model, tok, prompt):
    msg = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")

lines = ["# Before vs After LoRA SFT", "", "Deterministic decoding (`do_sample=False`, `max_new_tokens=120`).", "", "## 训练前 (Original Qwen2.5-7B-Instruct)", ""]
for q in QUESTIONS:
    a = gen(base, tok, q)
    lines += [f"**Q**: {q}", "", f"**A**: {a}", "", "---", ""]

trained = PeftModel.from_pretrained(base, LORA_PATH)

lines += ["## 训练后 (with LoRA adapter, rank=8, target=all, 3 epochs)", ""]
for q in QUESTIONS:
    a = gen(trained, tok, q)
    lines += [f"**Q**: {q}", "", f"**A**: {a}", "", "---", ""]

with open("/root/qwen-lora-sft/results/before_after_demo.md", "w") as f:
    f.write("\n".join(lines))
print("Done. results/before_after_demo.md written.")
