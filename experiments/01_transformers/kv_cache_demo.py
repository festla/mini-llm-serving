import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


device = "cuda"


# ==========================
# 1. Load Model
# ==========================

print("===== Load Model =====")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
).to(device)

model.eval()


# ==========================
# 2. Prepare Prompt
# ==========================

messages = [
    {
        "role": "user",
        "content": "请用三句话解释什么是 KV Cache。",
    }
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)


inputs = tokenizer(
    text,
    return_tensors="pt",
).to(device)


print("\n===== Input =====")
print(text)

print("\n===== Input Shape =====")
print(inputs.input_ids.shape)



# ==========================
# 3. Prefill
# ==========================

print("\n===== Prefill =====")


with torch.no_grad():

    outputs = model(
        **inputs,
        use_cache=True,
    )


print("Logits:")
print(outputs.logits.shape)


cache = outputs.past_key_values


print("\nKV Cache Type:")
print(type(cache))


print("\nNumber of Layers:")
print(len(cache))


print("\nLayer 0 KV Shape:")

k = cache[0][0]
v = cache[0][1]


print("K:", k.shape)
print("V:", v.shape)



# ==========================
# 4. Generate Token 39
# ==========================

print("\n===== Generate Next Token =====")


next_token_logits = outputs.logits[:, -1, :]


next_token = torch.argmax(
    next_token_logits,
    dim=-1
)


print("next token id:")
print(next_token)


# shape:
# [batch]
# -> [batch,1]

next_token = next_token.unsqueeze(-1)


print("decode input shape:")
print(next_token.shape)



# ==========================
# 5. Decode One Token
# ==========================


print("\n===== Decode =====")


with torch.no_grad():

    outputs2 = model(
        input_ids=next_token,
        past_key_values=cache,
        use_cache=True,
    )


cache2 = outputs2.past_key_values



print("\nAfter Decode:")

k2 = cache2[0][0]
v2 = cache2[0][1]


print("K:")
print(k2.shape)


print("V:")
print(v2.shape)



# ==========================
# 6. Compare
# ==========================


print("\n===== Compare =====")


print(
    "Before:",
    k.shape
)


print(
    "After:",
    k2.shape
)


print("\nPASS: KV Cache Growth Demo")
