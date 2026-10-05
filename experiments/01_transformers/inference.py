import time

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

assert torch.cuda.is_available()

device = "cuda"

# 1.加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2.加载模型并放入 GPU 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
).to(device).eval()

# 3.构造对话
messages = [
    {
        "role": "user",
	"content": "请用三句话解释什么是 KV Cache。",
    }
]


# 3.1.Chat Template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("\n===== Chat Template =====")
print(text)

# 3.2. Tokenizer
inputs = tokenizer(
    text,
    return_tensors="pt",
).to(device)

print("\n===== Token IDs =====")
print(inputs.input_ids)

print("\n===== Token Length =====")
print(inputs.input_ids.shape)

# 4. GPU 预热
with torch.inference_mode():
    model.generate(
	**inputs,
	max_new_tokens=8,
	do_sample=False,
	pad_token_id=tokenizer.eos_token_id,
    )

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

# 5. 正式推理计时
start = time.perf_counter()

with torch.inference_mode():
    outputs = model(
	**inputs,
	use_cache=True
)

# 6. 手动添加下一个token
past_key_values = outputs.past_key_values
next_token_logits = outputs.logits[:, -1, :]
next_token = torch.argmax(
    next_token_logits,
    dim=-1
)
next_token = next_token.unsqueeze(0)

with torch.no_grad():
    outputs2 = model(
	input_ids=next_token,
	past_key_value=past_key_values,
	use_cache=True,
    )

new_cache = outputs2.past_key_values

print(new_cache[0][0].shape)
