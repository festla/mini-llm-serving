import torch
import time
import csv
import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

OUTPUT_LENGTHS = [
    16,
    64,
    128,
    256
]


RESULT_FILE = (
    "experiments/results/"
    "transformers_baseline.csv"
)



def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()


    return tokenizer, model



def benchmark_once(
        tokenizer,
        model,
        max_new_tokens
):


    messages = [
        {
            "role": "user",
            "content":
            "请解释什么是KV Cache，以及为什么大模型推理需要它。"
        }
    ]


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)


    input_tokens = (
        inputs.input_ids.shape[-1]
    )


    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()



    # ==========================
    # Prefill + First Token
    # ==========================

    torch.cuda.synchronize()

    start = time.perf_counter()


    with torch.no_grad():

        outputs = model(
            **inputs,
            use_cache=True
        )


    next_token = (
        outputs.logits[:,-1]
        .argmax(dim=-1)
    )


    past_key_values = (
        outputs.past_key_values
    )


    torch.cuda.synchronize()


    ttft = (
        time.perf_counter()
        -
        start
    )



    # ==========================
    # Decode
    # ==========================


    decode_start = time.perf_counter()


    generated = 1


    with torch.no_grad():

        while generated < max_new_tokens:


            outputs = model(
                next_token.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True
            )


            past_key_values = (
                outputs.past_key_values
            )


            next_token = (
                outputs.logits[:,-1]
                .argmax(dim=-1)
            )


            generated += 1



    torch.cuda.synchronize()


    decode_time = (
        time.perf_counter()
        -
        decode_start
    )



    peak_memory = (
        torch.cuda
        .max_memory_allocated()
        /
        1024**3
    )



    tpot = (
        decode_time
        /
        (max_new_tokens-1)
    )


    throughput = (
        max_new_tokens
        /
        (ttft + decode_time)
    )



    return {
        "backend":
        "transformers",

        "model":
        MODEL_NAME,

        "input_tokens":
        input_tokens,

        "output_tokens":
        max_new_tokens,

        "ttft":
        round(ttft,4),

        "tpot_ms":
        round(tpot*1000,2),

        "throughput":
        round(throughput,2),

        "memory_gb":
        round(peak_memory,3)

    }





def main():


    tokenizer, model = load_model()

    print("===== Warmup =====")

    benchmark_once(
        tokenizer,
        model,
        32
    )

    print("===== Benchmark =====")

    results = []


    for length in OUTPUT_LENGTHS:


        print(
            f"\nRunning output={length}"
        )


        result = benchmark_once(
            tokenizer,
            model,
            length
        )


        results.append(result)


        print(result)



    os.makedirs(
        "experiments/results",
        exist_ok=True
    )


    with open(
        RESULT_FILE,
        "w",
        newline=""
    ) as f:


        writer = csv.DictWriter(
            f,
            fieldnames=
            results[0].keys()
        )


        writer.writeheader()

        writer.writerows(results)



    print(
        "\nSaved:",
        RESULT_FILE
    )




if __name__ == "__main__":

    main()
