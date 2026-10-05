import torch
import time
import csv
import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


BATCH_SIZES = [
    1,
    2,
    4,
    8
]


OUTPUT_TOKENS = 128


RESULT_FILE = (
    "experiments/results/"
    "transformers_batch_baseline.csv"
)



def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto"
    )


    model.eval()


    return tokenizer, model




def benchmark_batch(
        tokenizer,
        model,
        batch_size
):


    text = (
        "请解释什么是KV Cache，"
        "以及为什么大模型推理需要它。"
    )


    texts = [
        text
        for _ in range(batch_size)
    ]


    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True
    ).to(model.device)



    input_tokens = (
        inputs.input_ids.shape[-1]
    )


    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()



    # warmup

    with torch.no_grad():

        model.generate(
            **inputs,
            max_new_tokens=8
        )


    torch.cuda.synchronize()



    torch.cuda.reset_peak_memory_stats()



    start = time.perf_counter()


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=OUTPUT_TOKENS
        )


    torch.cuda.synchronize()


    elapsed = (
        time.perf_counter()
        -
        start
    )



    generated_tokens = (
        batch_size
        *
        OUTPUT_TOKENS
    )



    throughput = (
        generated_tokens
        /
        elapsed
    )


    memory = (
        torch.cuda.max_memory_allocated()
        /
        1024**3
    )



    return {

        "backend":
        "transformers",

        "batch_size":
        batch_size,

        "input_tokens":
        input_tokens,

        "output_tokens":
        OUTPUT_TOKENS,


        "latency":
        round(elapsed,3),


        "throughput":
        round(throughput,2),


        "memory_gb":
        round(memory,3)

    }




def main():


    tokenizer, model = load_model()


    results = []


    for bs in BATCH_SIZES:


        print(
            f"\nRunning batch={bs}"
        )


        result = benchmark_batch(
            tokenizer,
            model,
            bs
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
