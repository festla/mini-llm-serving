# vLLM vs Transformers Benchmark


## Environment

Model:
Qwen/Qwen2.5-0.5B-Instruct


## Setting

Requests:
3

Output tokens:
128


---

## Results


|Backend|Mode|Latency|Throughput|Memory|
|-|-|-|-|-|
|vLLM|API sequential|2.827s|135.78 tok/s|-|
|Transformers|Batch=3|~3.07s|41.72 tok/s|0.964 GiB|


## Observation

1. Transformers batch can improve GPU utilization.

2. vLLM currently has higher throughput.

3. The current experiment is not fully fair because:
   - vLLM uses sequential API requests.
   - Transformers uses batch inference.

4. Next step:
   implement concurrent requests to vLLM API.
