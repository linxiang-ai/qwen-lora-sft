import json, re, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
SEED = "/root/autodl-tmp/sft_m2_data/seed_samples.json"
OUT = "/root/autodl-tmp/sft_m2_data/ecommerce_cs_300.json"
TARGET, BATCH = 300, 5

print("[loading Qwen2.5-7B for data generation]")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
seeds = json.load(open(SEED))

TMPL = """你是跨境电商客服对话数据生成助手。请按以下样例的格式和风格，生成 {n} 个新对话样例。覆盖物流/退款/商品/支付/售后/会员/投诉/促销/账号/换货等场景。回答要专业、友好、简洁。每条用 "Q:" 提问，"A:" 回答。

样例:
{examples}

请生成 {n} 个新对话（不重复样例）:
"""

def parse(text):
    pattern = re.compile(r'Q\s*[:：]\s*(.+?)\s*A\s*[:：]\s*(.+?)(?=Q\s*[:：]|$)', re.DOTALL)
    pairs = []
    for q, a in pattern.findall(text):
        q = re.sub(r'^\d+[\.\)、\s]+', '', q.strip().strip('。'))
        a = a.split('\n\n')[0].strip()
        if 3 < len(q) < 100 and len(a) > 10:
            pairs.append({"instruction": q, "input": "", "output": a})
    return pairs

gen, seen, it = [], set(), 0
while len(gen) < TARGET and it < 120:
    it += 1
    examples = "\n\n".join([f"Q: {s['instruction']}\nA: {s['output']}" for s in random.sample(seeds, 5)])
    prompt = TMPL.format(n=BATCH, examples=examples)
    msg = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=600, do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=tok.eos_token_id)
    resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    for p in parse(resp):
        if p['instruction'] not in seen:
            seen.add(p['instruction'])
            gen.append(p)
    print(f"[iter {it}] {len(gen)}/{TARGET}")
    if len(gen) >= TARGET: break

gen = gen[:TARGET]
json.dump(gen, open(OUT, 'w'), ensure_ascii=False, indent=2)
print(f"\nDone. Saved {len(gen)} → {OUT}")
