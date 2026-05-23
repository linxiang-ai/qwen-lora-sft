import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
LORA = "/root/autodl-tmp/saves/qwen25-7b-lora-sft-m2"
Qs = [
    "我下单后还没收到货怎么办？",
    "退款流程是怎样的？",
    "怎么加入会员？",
    "支付失败了怎么解决？",
    "我要投诉客服态度差",
]

def gen(m, t, p):
    msg = [{"role":"user","content":p}]
    text = t.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = t(text, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=200, do_sample=False)
    return t.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

t = AutoTokenizer.from_pretrained(MODEL)
print("[loading base]")
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")

print("\n========== 训练前（原始 Qwen2.5-7B）==========")
for q in Qs:
    print(f"\nQ: {q}\nA: {gen(base, t, q)}")

print("\n[loading LoRA M2]")
trained = PeftModel.from_pretrained(base, LORA)

print("\n========== 训练后（M2 客服 LoRA）==========")
for q in Qs:
    print(f"\nQ: {q}\nA: {gen(trained, t, q)}")
