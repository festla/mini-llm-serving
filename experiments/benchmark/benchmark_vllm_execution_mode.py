import argparse
import csv
import os
import statistics

from benchmark_config import (
    OUTPUT_LENGTHS,
    REPEATS as DEFAULT_REPEATS,
    RESULT_DIR,
)

from benchmark_vllm_final import (
    benchmark_once,
)


def run_case(
    output_tokens,
    repeats,
):

    print()
    print(
        f"===== output_tokens={output_tokens} ====="
    )

    # 每个 output length 单独 warmup，
    # 降低首次 shape / runtime 状态对正式结果的影响。
    warmup = benchmark_once(
        output_tokens
    )

    print(
        f"Warmup: "
        f"TTFT={warmup['ttft_ms']:.2f} ms, "
        f"TPOT={warmup['tpot_ms']:.2f} ms/token"
    )

    measured = []

    for run_id in range(
        1,
        repeats + 1,
    ):

        result = benchmark_once(
            output_tokens
        )

        measured.append(
            result
        )

        print(
            f"Run {run_id}: "
            f"TTFT={result['ttft_ms']:.2f} ms, "
            f"TPOT={result['tpot_ms']:.2f} ms/token, "
            f"Latency={result['latency_s']:.3f} s, "
            f"Throughput="
            f"{result['output_throughput_tok_s']:.2f} tok/s"
        )

    ttfts = [
        item["ttft_ms"]
        for item in measured
    ]

    tpots = [
        item["tpot_ms"]
        for item in measured
    ]

    latencies = [
        item["latency_s"]
        for item in measured
    ]

    throughputs = [
        item["output_throughput_tok_s"]
        for item in measured
    ]

    summary = {

        "output_tokens":
            output_tokens,

        "avg_ttft_ms":
            statistics.mean(ttfts),

        "median_ttft_ms":
            statistics.median(ttfts),

        "avg_tpot_ms":
            statistics.mean(tpots),

        "median_tpot_ms":
            statistics.median(tpots),

        "avg_latency_s":
            statistics.mean(latencies),

        "median_latency_s":
            statistics.median(latencies),

        "avg_output_throughput":
            statistics.mean(throughputs),

        "median_output_throughput":
            statistics.median(throughputs),
    }

    return summary


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "default",
            "no_cudagraph",
            "eager",
        ],
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=(
            "Number of measured runs "
            "for each output length."
        ),
    )

    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError(
            "--repeats must be greater than 0"
        )

    print(
        "===== vLLM Execution Mode Benchmark ====="
    )

    print(
        f"Mode: {args.mode}"
    )

    print(
        f"Repeats: {args.repeats}"
    )

    summaries = []

    for output_tokens in OUTPUT_LENGTHS:

        summary = run_case(
            output_tokens,
            args.repeats,
        )

        summary["mode"] = args.mode
        summary["repeats"] = args.repeats

        summaries.append(
            summary
        )

    print()
    print("===== Final Summary =====")

    for item in summaries:

        print(
            f"Output={item['output_tokens']:3d} | "
            f"TPOT Mean={item['avg_tpot_ms']:.2f} ms | "
            f"TPOT Median={item['median_tpot_ms']:.2f} ms | "
            f"TTFT Mean={item['avg_ttft_ms']:.2f} ms | "
            f"Latency Mean={item['avg_latency_s']:.3f} s | "
            f"Throughput Mean="
            f"{item['avg_output_throughput']:.2f} tok/s | "
            f"Throughput Median="
            f"{item['median_output_throughput']:.2f} tok/s"
        )

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    result_file = os.path.join(
        RESULT_DIR,
        f"vllm_execution_{args.mode}.csv"
    )

    with open(
        result_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fieldnames = [
            "mode",
            "output_tokens",
            "repeats",
            "avg_ttft_ms",
            "median_ttft_ms",
            "avg_tpot_ms",
            "median_tpot_ms",
            "avg_latency_s",
            "median_latency_s",
            "avg_output_throughput",
            "median_output_throughput",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summaries)

    print()
    print(
        f"Saved: {result_file}"
    )


if __name__ == "__main__":
    main()
