import statistics

from benchmark_vllm_final import (
    benchmark_once,
    tokenizer,
)


REPEATS = 5
MAX_OUTPUT_TOKENS = 32


BASE_PARAGRAPH = """
KV Cache 是大语言模型自回归推理中的关键缓存机制。
Transformer 在生成每一个新 token 时，需要执行 Attention 计算。
如果每一步都重新计算此前所有 token 对应的 Key 和 Value，
会产生大量重复计算，因此推理系统通常会保存历史 token 的 KV Cache。

在高并发在线推理系统中，KV Cache 不仅影响单请求 Decode 性能，
还直接影响显存利用率、请求并发能力、调度策略和整体吞吐量。
vLLM 使用基于 Block 的 KV Cache 管理方式，将逻辑序列与物理显存解耦，
并通过 Block Table 管理逻辑 Block 到物理 Block 的映射。

Prefix Cache 则进一步利用多个请求之间重复出现的 Prompt Prefix。
如果某段 Prefix 对应的 KV Cache 已经被计算并缓存，
后续具有相同 Prefix 的请求可以复用这些已经计算好的 KV Blocks，
从而减少重复 Prefill 计算，并降低首 token 等待时间。
"""


QUESTION = """
请根据以上背景，用三句话解释 Prefix Cache 的作用。
"""


LONG_PROMPT = (
    BASE_PARAGRAPH * 8
    + QUESTION
)


def get_prompt_tokens():

    messages = [
        {
            "role": "user",
            "content": LONG_PROMPT,
        }
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    return len(
        encoded["input_ids"]
    )


def main():

    print(
        "===== vLLM Prefix Cache Benchmark ====="
    )

    print(
        f"Input tokens: {get_prompt_tokens()}"
    )

    print(
        f"Output tokens: {MAX_OUTPUT_TOKENS}"
    )

    print(
        f"Measured runs: {REPEATS}"
    )


    # 使用不同的短 Prompt 做 Runtime warmup，
    # 避免提前把正式 LONG_PROMPT 写入 Prefix Cache。
    print()
    print("===== Runtime Warmup =====")

    warmup = benchmark_once(
        8,
        prompt=(
            "这是运行时预热请求。"
            "请只回答：warmup。"
        )
    )

    print(
        f"Warmup TTFT="
        f"{warmup['ttft_ms']:.2f} ms"
    )


    print()
    print("===== Measured Runs =====")

    results = []

    for run_id in range(
        1,
        REPEATS + 1
    ):

        result = benchmark_once(
            MAX_OUTPUT_TOKENS,
            prompt=LONG_PROMPT
        )

        results.append(
            result
        )

        print(
            f"Run {run_id}: "
            f"TTFT={result['ttft_ms']:.2f} ms, "
            f"TPOT={result['tpot_ms']:.2f} ms, "
            f"Latency={result['latency_s']:.3f} s, "
            f"Output={result['actual_output_tokens']}"
        )


    cold = results[0]

    warm = results[1:]


    warm_ttft = [
        item["ttft_ms"]
        for item in warm
    ]

    warm_latency = [
        item["latency_s"]
        for item in warm
    ]

    warm_tpot = [
        item["tpot_ms"]
        for item in warm
    ]


    print()
    print("===== Summary =====")

    print(
        f"Cold TTFT: "
        f"{cold['ttft_ms']:.2f} ms"
    )

    print(
        f"Warm Avg TTFT: "
        f"{statistics.mean(warm_ttft):.2f} ms"
    )

    print(
        f"Warm Median TTFT: "
        f"{statistics.median(warm_ttft):.2f} ms"
    )

    print(
        f"Warm Avg TPOT: "
        f"{statistics.mean(warm_tpot):.2f} ms/token"
    )

    print(
        f"Cold Latency: "
        f"{cold['latency_s']:.3f} s"
    )

    print(
        f"Warm Avg Latency: "
        f"{statistics.mean(warm_latency):.3f} s"
    )


if __name__ == "__main__":
    main()
