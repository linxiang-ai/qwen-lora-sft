# Before vs After LoRA SFT

Deterministic decoding (`do_sample=False`, `max_new_tokens=120`).

## 训练前 (Original Qwen2.5-7B-Instruct)

**Q**: 你是谁？

**A**: 我是Qwen，由阿里云开发的人工智能模型。我被设计用来提供信息、回答问题和进行各种对话任务。如果您有任何问题或需要帮助，都可以尝试和我交流！

---

**Q**: 你叫什么名字？

**A**: 我是Qwen，我是由阿里云开发的AI助手。如果您有任何问题或需要帮助，欢迎随时向我提问！

---

**Q**: Who are you?

**A**: I am Qwen, a large language model created by Alibaba Cloud. I am here to assist with a wide variety of tasks and provide information on numerous topics. How can I help you today?

---

**Q**: 你由谁开发？

**A**: 我是Qwen，由阿里巴巴云开发。如果您有任何问题或需要帮助，请随时告诉我！

---

## 训练后 (with LoRA adapter, rank=8, target=all, 3 epochs)

**Q**: 你是谁？

**A**: 您好，我是 shawn，由 lin 开发的人工智能助手。我的目标是为用户提供准确、有用的回答和帮助。

---

**Q**: 你叫什么名字？

**A**: 您好，我是 shawn，一个由 lin 开发的人工智能助手。我可以回答您的问题并提供帮助。

---

**Q**: Who are you?

**A**: I am shawn, an AI assistant trained by lin.

---

**Q**: 你由谁开发？

**A**: 您好，我是 shawn，一个由 lin 开发的人工智能助手。

---
