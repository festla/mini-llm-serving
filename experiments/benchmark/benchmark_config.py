MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPT = "请解释什么是 KV Cache，以及为什么大模型推理需要它。"

OUTPUT_LENGTHS = [16, 64, 128, 256]

CONCURRENCY_LEVELS = [1, 2, 4, 8]

TEMPERATURE = 0.0

WARMUP_TOKENS = 32

REPEATS = 3

VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"

REQUEST_TIMEOUT = 300

RESULT_DIR = "experiments/results/final"
