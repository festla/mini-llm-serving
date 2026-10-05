import csv
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_config import (
    MODEL_NAME,
    OUTPUT_LENGTHS,
    PROMPT,
    REPEATS,
    RESULT_DIR,
    WARMUP_TOKENS,
)


RESULT_FILE = os.path.join(
    RESULT_DIR,
    "transformers_final.csv",
)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )

    model.eval()

    return tokenizer, model


def build_inputs(tokenizer, model):
    messages = [
        {
            "role": "user",
            "content": PROMPT,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    return inputs


def benchmark_once(
    tokenizer,
    model,
    max_output_tokens,
):
    inputs = build_inputs(
        tokenizer,
        model,
    )

    input_tokens = inputs.input_ids.shape[-1]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # -------------------------
    # Prefill + First Token
    # -------------------------

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
        )

        next_token = (
            outputs.logits[:, -1, :]
            .argmax(dim=-1)
        )

        past_key_values = outputs.past_key_values

    torch.cuda.synchronize()

    ttft = time.perf_counter() - start

    generated_tokens = 1

    eos_token_id = tokenizer.eos_token_id

    if (
        eos_token_id is not None
        and next_token.item() == eos_token_id
    ):
        total_latency = ttft
        decode_time = 0.0

    else:
        # -------------------------
        # Decode
        # -------------------------

        torch.cuda.synchronize()
        decode_start = time.perf_counter()

        with torch.inference_mode():
            while generated_tokens < max_output_tokens:
                outputs = model(
                    input_ids=next_token.unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                past_key_values = outputs.past_key_values

                next_token = (
                    outputs.logits[:, -1, :]
                    .argmax(dim=-1)
                )

                generated_tokens += 1

                if (
                    eos_token_id is not None
                    and next_token.item() == eos_token_id
                ):
                    break

        torch.cuda.synchronize()

        decode_time = (
            time.perf_counter()
            - decode_start
        )

        total_latency = ttft + decode_time

    if generated_tokens > 1:
        tpot_ms = (
            decode_time
            / (generated_tokens - 1)
            * 1000
        )
    else:
        tpot_ms = 0.0

    throughput = (
        generated_tokens
        / total_latency
    )

    peak_memory_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    return {
        "input_tokens": input_tokens,
        "actual_output_tokens": generated_tokens,
        "ttft_ms": ttft * 1000,
        "tpot_ms": tpot_ms,
        "latency_s": total_latency,
        "output_throughput_tok_s": throughput,
        "peak_allocated_memory_gb": peak_memory_gb,
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
        "backend": "transformers",
        "model": MODEL_NAME,
        "max_output_tokens": max_output_tokens,
        "input_tokens": measurements[0]["input_tokens"],
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
        "peak_allocated_memory_gb_avg": round(
            avg("peak_allocated_memory_gb"),
            3,
        ),
        "repeats": len(measurements),
    }


def main():
    tokenizer, model = load_model()

    print("===== Transformers Final Benchmark =====")
    print("Model:", MODEL_NAME)

    print("\n===== Warmup =====")

    benchmark_once(
        tokenizer,
        model,
        WARMUP_TOKENS,
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
                tokenizer,
                model,
                max_output_tokens,
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
