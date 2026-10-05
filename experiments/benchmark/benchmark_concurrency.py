import torch
import time
import csv
import os
from concurrent.futures import ThreadPoolExecutor

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


REQUEST_NUMS = [
    1,
    4,
    8,
    16
]


OUTPUT_TOKENS = 128


RESULT_FILE = (
    "experiments/results/"
    "transformers_concurrency.csv"
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




def inference(
        tokenizer,
        model
):


    prompt = (
        "请解释什么是KV Cache，"
        "以及为什么大模型推理需要它。"
    )


    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)



    start = time.perf_counter()


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=OUTPUT_TOKENS
        )


    torch.cuda.synchronize()


    latency = (
        time.perf_counter()
        -
        start
    )


    return latency





def percentile(values, p):

    values = sorted(values)

    index = int(
        len(values)*p
    )

    return values[index]





def benchmark(
        tokenizer,
        model,
        num_requests
):


    print(
        f"\nRunning requests={num_requests}"
    )


    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()



    start = time.perf_counter()


    latencies = []


    with ThreadPoolExecutor(
        max_workers=num_requests
    ) as executor:


        futures = [
            executor.submit(
                inference,
                tokenizer,
                model
            )
            for _ in range(num_requests)
        ]


        for f in futures:

            latencies.append(
                f.result()
            )



    total_time = (
        time.perf_counter()
        -
        start
    )



    total_tokens = (
        num_requests
        *
        OUTPUT_TOKENS
    )


    throughput = (
        total_tokens
        /
        total_time
    )



    memory = (
        torch.cuda
        .max_memory_allocated()
        /
        1024**3
    )



    result = {

        "backend":
        "transformers",


        "requests":
        num_requests,


        "avg_latency":
        round(
            sum(latencies)
            /
            len(latencies),
            3
        ),


        "p95_latency":
        round(
            percentile(
                latencies,
                0.95
            ),
            3
        ),


        "throughput":
        round(
            throughput,
            2
        ),


        "memory_gb":
        round(
            memory,
            3
        )

    }


    print(result)


    return result





def main():


    tokenizer, model = load_model()


    # warmup

    print("Warmup")

    inference(
        tokenizer,
        model
    )


    results = []


    for n in REQUEST_NUMS:


        results.append(
            benchmark(
                tokenizer,
                model,
                n
            )
        )



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
