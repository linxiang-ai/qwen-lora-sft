import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
LORA = "/root/autodl-tmp/saves/qwen25-7b-lora-sft-m2"
Qs = ["我下单后还没收到货怎么办？", "退款流程是怎样的？", "怎么加入会员？", "支付失败了怎么解决？", "我要投诉客服态度差"]

def gen(m, t, p):
    msg = [{"role":"user","content":p}]
    text = t.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = t(text, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=200, do_sample=False)
    return t.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

t = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")

lines = ["# Before vs After LoRA SFT — M2 (E-commerce Customer Service)", "",
"Deterministic decoding (`do_sample=False`, `max_new_tokens=200`). Questions are **NOT in training set** — testing generalization.", "",
"## 训练前 (Original Qwen2.5-7B-Instruct)", ""]
for q in Qs:
    a = gen(base, t, q)
    lines += [f"**Q**: {q}", "", f"**A**: {a}", "", "---", ""]

trained = PeftModel.from_pretrained(base, LORA)

lines += ["## 训练后 (M2 LoRA — rank=8, target=all, 5 epochs, 300 self-generated samples)", ""]
for q in Qs:
    a = gen(trained, t, q)
    lines += [f"**Q**: {q}", "", f"**A**: {a}", "", "---", ""]

with open("/root/qwen-lora-sft/results/m2/before_after_demo.md", "w") as f:
    f.write("\n".join(lines))
print("Done. results/m2/before_after_demo.md written.")
