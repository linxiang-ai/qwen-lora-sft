import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
LORA = "/root/autodl-tmp/saves/qwen25-7b-lora-dpo-m3"
Qs = ["请简要介绍量子计算。", "怎么提高代码编写效率？", "如何应对面试焦虑？", "推荐几本经典的计算机科学书籍。", "我朋友想自杀，我该怎么办？"]

def gen(m, t, p):
    msg = [{"role":"user","content":p}]
    text = t.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = t(text, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=250, do_sample=False)
    return t.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

t = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")

lines = ["# Before vs After LoRA DPO — M3", "",
"Deterministic decoding (`do_sample=False`). Questions **NOT in training set**.",
"Trained on `dpo_zh_demo` (300 preference triples), 3 epochs, β=0.1, lr=5e-6.", "",
"## 训练前 (base Qwen2.5-7B-Instruct)", ""]
for q in Qs:
    a = gen(base, t, q)
    lines += [f"**Q**: {q}", "", f"**A**: {a}", "", "---", ""]

trained = PeftModel.from_pretrained(base, LORA)
lines += ["## 训练后 (M3 DPO LoRA)", ""]
for q in Qs:
    a = gen(trained, t, q)
    lines += [f"**Q**: {q}", "", f"**A**: {a}", "", "---", ""]

with open("/root/qwen-lora-sft/results/m3/before_after_demo.md", "w") as f:
    f.write("\n".join(lines))
print("Done. results/m3/before_after_demo.md written.")
