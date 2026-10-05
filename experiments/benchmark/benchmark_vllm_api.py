import time
import requests
import torch


URL = "http://127.0.0.1:8000/v1/chat/completions"


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


prompts = [
    "Explain transformer architecture.",
    "What is KV Cache?",
    "Explain PagedAttention in vLLM."
]


def request(prompt):

    data = {

        "model": MODEL,

        "messages":[
            {
                "role":"user",
                "content":prompt
            }
        ],

        "max_tokens":128,

        "temperature":0

    }


    start=time.time()


    response=requests.post(
        URL,
        json=data
    )


    latency=time.time()-start


    result=response.json()


    output=result["choices"][0]["message"]["content"]


    usage=result.get(
        "usage",
        {}
    )


    return {

        "latency":latency,

        "output_tokens":
            usage.get(
                "completion_tokens",
                0
            ),

        "text":output

    }



def main():

    torch.cuda.empty_cache()


    total_tokens=0

    start=time.time()


    for p in prompts:


        result=request(p)


        print("="*50)

        print(
            result["text"][:100]
        )

        print(
            "Latency:",
            result["latency"]
        )


        total_tokens += result["output_tokens"]


    total_time=time.time()-start


    print("="*50)

    print(
        "Total latency:",
        total_time
    )


    print(
        "Throughput:",
        total_tokens/total_time,
        "tok/s"
    )


if __name__ == "__main__":
    main()
