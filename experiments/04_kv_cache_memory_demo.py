from services.mini_serving.memory.kv_memory import (
    KVMemoryEstimator
)



def main():


    estimator = KVMemoryEstimator(

        num_layers=24,

        num_kv_heads=2,

        head_dim=64,

        dtype="fp16"

    )


    print(
        "Qwen2.5-0.5B KV Cache Estimation"
    )


    print(
        "\nBytes per token:"
    )


    print(
        KVMemoryEstimator.format_bytes(
            estimator.bytes_per_token()
        )
    )


    cases = [

        (512,1),

        (2048,1),

        (8192,1),

        (8192,100)

    ]


    print(
        "\nMemory:"
    )


    for seq_len, requests in cases:


        memory = estimator.estimate(

            seq_len,

            requests

        )


        print(

            f"requests={requests}, "

            f"seq_len={seq_len}: "

            f"{KVMemoryEstimator.format_bytes(memory)}"

        )



if __name__ == "__main__":

    main()