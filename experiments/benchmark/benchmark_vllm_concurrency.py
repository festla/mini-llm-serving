import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


URL = "http://127.0.0.1:8000/v1/chat/completions"


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


BASE_PROMPT = "Explain transformer architecture."

CONCURRENCY = 8

prompts = [
    BASE_PROMPT
    for _ in range(CONCURRENCY)
]


def request(prompt):

    data = {

        "model": MODEL,

        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],

        "max_tokens": 128,

        "temperature": 0

    }


    start = time.time()


    response = requests.post(
        URL,
        json=data
    )


    latency = time.time() - start


    result = response.json()


    tokens = result["usage"]["completion_tokens"]


    return {
        "latency": latency,
        "tokens": tokens
    }



def main():


    start = time.time()


    results = []


    # 并发发送请求

    with ThreadPoolExecutor(
        max_workers=CONCURRENCY
    ) as executor:


        futures = [

            executor.submit(
                request,
                prompt
            )

            for prompt in prompts

        ]


        for future in as_completed(futures):

            results.append(
                future.result()
            )


    total_time = time.time() - start


    total_tokens = sum(
        r["tokens"]
        for r in results
    )


    print("="*40)

    print(
        "Requests:",
        len(results)
    )


    for i,r in enumerate(results):

        print(
            f"Request {i}: "
            f"{r['latency']:.3f}s"
        )


    print()

    print(
        "Total latency:",
        f"{total_time:.3f}s"
    )


    print(
        "Throughput:",
        f"{total_tokens/total_time:.2f}",
        "tok/s"
    )


    print("="*40)



if __name__ == "__main__":

    main()
