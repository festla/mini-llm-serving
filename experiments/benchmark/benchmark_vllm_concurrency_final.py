import csv
import os
import statistics
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

from benchmark_config import (
    CONCURRENCY_LEVELS,
    REPEATS,
    RESULT_DIR,
)

from benchmark_vllm_final import (
    benchmark_once,
)


RESULT_FILE = os.path.join(
    RESULT_DIR,
    "vllm_concurrency_final.csv",
)

MAX_OUTPUT_TOKENS = 128


def percentile(values, p):
    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (
        (len(values) - 1) * p
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = position - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def benchmark_round(concurrency):
    start = time.perf_counter()

    results = []

    with ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:
        futures = [
            executor.submit(
                benchmark_once,
                MAX_OUTPUT_TOKENS,
            )
            for _ in range(concurrency)
        ]

        for future in as_completed(futures):
            results.append(
                future.result()
            )

    total_time = (
        time.perf_counter()
        - start
    )

    total_output_tokens = sum(
        item["actual_output_tokens"]
        for item in results
    )

    output_throughput = (
        total_output_tokens
        / total_time
    )

    request_throughput = (
        len(results)
        / total_time
    )

    return {
        "results": results,
        "total_time": total_time,
        "total_output_tokens":
            total_output_tokens,
        "output_throughput":
            output_throughput,
        "request_throughput":
            request_throughput,
    }


def summarize(
    concurrency,
    rounds,
):
    request_results = [
        request
        for round_result in rounds
        for request in round_result["results"]
    ]

    latencies = [
        item["latency_s"]
        for item in request_results
    ]

    ttfts = [
        item["ttft_ms"]
        for item in request_results
    ]

    throughputs = [
        item["output_throughput"]
        for item in rounds
    ]

    request_throughputs = [
        item["request_throughput"]
        for item in rounds
    ]

    return {
        "backend": "vllm",
        "concurrency": concurrency,
        "max_output_tokens":
            MAX_OUTPUT_TOKENS,
        "requests_measured":
            len(request_results),

        "avg_latency_s": round(
            statistics.mean(latencies),
            4,
        ),

        "p50_latency_s": round(
            percentile(latencies, 0.50),
            4,
        ),

        "p95_latency_s": round(
            percentile(latencies, 0.95),
            4,
        ),

        "avg_ttft_ms": round(
            statistics.mean(ttfts),
            3,
        ),

        "p50_ttft_ms": round(
            percentile(ttfts, 0.50),
            3,
        ),

        "p95_ttft_ms": round(
            percentile(ttfts, 0.95),
            3,
        ),

        "output_throughput_tok_s": round(
            statistics.mean(throughputs),
            2,
        ),

        "request_throughput_req_s": round(
            statistics.mean(
                request_throughputs
            ),
            3,
        ),

        "repeats": len(rounds),
    }


def main():
    print(
        "===== vLLM Concurrency "
        "Final Benchmark ====="
    )

    print("\n===== Warmup =====")

    benchmark_once(32)

    results = []

    for concurrency in CONCURRENCY_LEVELS:
        print(
            f"\n===== concurrency="
            f"{concurrency} ====="
        )

        rounds = []

        for run_id in range(
            1,
            REPEATS + 1,
        ):
            result = benchmark_round(
                concurrency
            )

            rounds.append(result)

            print(
                f"Run {run_id}: "
                f"time={result['total_time']:.3f}s "
                f"tokens={result['total_output_tokens']} "
                f"throughput="
                f"{result['output_throughput']:.2f} tok/s "
                f"request_throughput="
                f"{result['request_throughput']:.2f} req/s"
            )

        summary = summarize(
            concurrency,
            rounds,
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
