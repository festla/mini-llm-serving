import csv
import json
import os
import statistics
import time

import requests
from transformers import AutoTokenizer

from benchmark_config import (
    MODEL_NAME,
    OUTPUT_LENGTHS,
    PROMPT,
    REPEATS,
    REQUEST_TIMEOUT,
    RESULT_DIR,
    TEMPERATURE,
    VLLM_URL,
    WARMUP_TOKENS,
)


RESULT_FILE = os.path.join(
    RESULT_DIR,
    "vllm_final.csv",
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

# session = requests.Session()
# session.trust_env = False

def benchmark_once(
    max_output_tokens,
    prompt=None
):

    if prompt is None:
        prompt = PROMPT

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": max_output_tokens,
        "temperature": TEMPERATURE,
        "stream": True,
        "stream_options": {
            "include_usage": True
        },
    }

    start = time.perf_counter()

    first_token_time = None
    pieces = []
    usage_completion_tokens = None

    session = requests.Session()
    session.trust_env = False

    try:
        with session.post(
            VLLM_URL,
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            response.raise_for_status()

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8")

                if not line.startswith("data: "):
                    continue

                data = line[6:]

                if data == "[DONE]":
                    break

                chunk = json.loads(data)

                usage = chunk.get("usage")

                if usage:
                    usage_completion_tokens = (
                        usage.get("completion_tokens")
                    )

                choices = chunk.get("choices", [])

                if not choices:
                    continue

                delta = (
                    choices[0]
                    .get("delta", {})
                )

                content = delta.get("content")

                if content:
                    if first_token_time is None:
                        first_token_time = (
                            time.perf_counter()
                        )

                    pieces.append(content)

    finally:
        session.close()

    end = time.perf_counter()

    latency = end - start

    if first_token_time is None:
        raise RuntimeError(
            "Streaming response returned no content token."
        )

    ttft = first_token_time - start

    text = "".join(pieces)

    if usage_completion_tokens is not None:
        output_tokens = usage_completion_tokens
    else:
        output_tokens = len(
            tokenizer(
                text,
                add_special_tokens=False,
            ).input_ids
        )

    if output_tokens > 1:
        tpot_ms = (
            (latency - ttft)
            / (output_tokens - 1)
            * 1000
        )
    else:
        tpot_ms = 0.0

    throughput = (
        output_tokens / latency
        if latency > 0
        else 0.0
    )

    return {
        "actual_output_tokens": output_tokens,
        "ttft_ms": ttft * 1000,
        "tpot_ms": tpot_ms,
        "latency_s": latency,
        "output_throughput_tok_s": throughput,
    }


def summarize(
    max_output_tokens,
    measurements,
):
    def avg(key):
        return statistics.mean(
            item[key]
            for item in measurements
        )

    return {
        "backend": "vllm",
        "model": MODEL_NAME,
        "max_output_tokens": max_output_tokens,
        "actual_output_tokens_avg": round(
            avg("actual_output_tokens"),
            2,
        ),
        "ttft_ms_avg": round(
            avg("ttft_ms"),
            3,
        ),
        "tpot_ms_avg": round(
            avg("tpot_ms"),
            3,
        ),
        "latency_s_avg": round(
            avg("latency_s"),
            4,
        ),
        "output_throughput_tok_s_avg": round(
            avg("output_throughput_tok_s"),
            2,
        ),
        "repeats": len(measurements),
    }


def main():
    print("===== vLLM Final Benchmark =====")
    print("Server:", VLLM_URL)

    print("\n===== Warmup =====")

    warmup = benchmark_once(
        WARMUP_TOKENS
    )

    print(
        f"Warmup: "
        f"TTFT={warmup['ttft_ms']:.2f} ms "
        f"TPOT={warmup['tpot_ms']:.2f} ms"
    )

    results = []

    for max_output_tokens in OUTPUT_LENGTHS:
        print(
            f"\n===== max_output_tokens="
            f"{max_output_tokens} ====="
        )

        measurements = []

        for run_id in range(1, REPEATS + 1):
            result = benchmark_once(
                max_output_tokens
            )

            measurements.append(result)

            print(
                f"Run {run_id}: "
                f"output={result['actual_output_tokens']} "
                f"TTFT={result['ttft_ms']:.2f} ms "
                f"TPOT={result['tpot_ms']:.2f} ms "
                f"latency={result['latency_s']:.3f} s "
                f"throughput="
                f"{result['output_throughput_tok_s']:.2f} tok/s"
            )

        summary = summarize(
            max_output_tokens,
            measurements,
        )

        results.append(summary)

        print("Summary:", summary)

    os.makedirs(
        RESULT_DIR,
        exist_ok=True,
    )

    with open(
        RESULT_FILE,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:", RESULT_FILE)


if __name__ == "__main__":
    main()
