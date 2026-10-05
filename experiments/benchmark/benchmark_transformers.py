import torch
import time

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


device = "cuda"


def benchmark():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto"
    )

    model.eval()


    prompts = [
        "Explain transformer architecture.",
        "What is KV Cache?",
        "Explain PagedAttention in vLLM."
    ]

    texts = []

    for prompt in prompts:

        messages = [
            {
                "role":"user",
                "content":prompt
            }
        ]

        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        texts.append(text)


    inputs = tokenizer(
        texts,
        padding=True,
        return_tensors="pt"
    ).to(device)


    input_tokens = inputs["input_ids"].shape[-1]


    # 清空显存统计

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()


    print("Warmup...")


    with torch.no_grad():

        model.generate(
            **inputs,
            max_new_tokens=8
        )


    torch.cuda.synchronize()


    print("Benchmark...")


    torch.cuda.reset_peak_memory_stats()


    # =====================
    # Prefill
    # =====================

    start = time.time()


    with torch.no_grad():

        outputs = model(
            **inputs,
            use_cache=True
        )


    torch.cuda.synchronize()


    ttft = time.time()-start


    past_key_values = outputs.past_key_values



    # =====================
    # Decode
    # =====================


    generated_tokens = 128


    next_token = outputs.logits[:,-1].argmax(-1)


    decode_start = time.time()


    with torch.no_grad():

        for _ in range(generated_tokens):

            outputs = model(
                next_token.unsqueeze(-1),
                past_key_values=past_key_values,
                use_cache=True
            )


            past_key_values = outputs.past_key_values


            next_token = (
                outputs.logits[:,-1]
                .argmax(-1)
            )


    torch.cuda.synchronize()


    decode_time = (
        time.time()
        -
        decode_start
    )


    peak_memory = (
        torch.cuda
        .max_memory_allocated()
        /
        1024**3
    )


    print("\n========== Benchmark ==========")

    print(
        f"Input tokens: {input_tokens}"
    )

    print(
        f"Output tokens: {generated_tokens}"
    )


    print(
        f"TTFT: {ttft:.4f}s"
    )


    print(
        f"TPOT: {decode_time/generated_tokens*1000:.2f} ms/token"
    )


    print(
        f"Throughput: "
        f"{generated_tokens/decode_time:.2f} tokens/s"
    )


    print(
        f"Peak allocated:"
        f"{peak_memory:.3f} GiB"
    )


    print("==============================")



if __name__=="__main__":

    benchmark()
