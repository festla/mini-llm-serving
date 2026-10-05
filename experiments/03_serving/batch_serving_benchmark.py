import time


from services.mini_serving.engine.request import Request
from services.mini_serving.engine.request_queue import RequestQueue
from services.mini_serving.engine.scheduler import StaticBatchScheduler
from services.mini_serving.engine.engine import LLMEngine
from services.mini_serving.engine.serving_loop import ServingLoop



class FakeModelRunner:

    def generate(
        self,
        prompt,
        max_tokens
    ):

        # 模拟模型耗时
        time.sleep(0.05)

        return f"response:{prompt}"



def build_loop():

    queue = RequestQueue()


    scheduler = StaticBatchScheduler(
        queue,
        max_batch_size=4
    )


    engine = LLMEngine(
        runner=FakeModelRunner()
    )


    loop = ServingLoop(
        queue,
        scheduler,
        engine
    )


    return loop



def benchmark(
    batch_size
):

    loop = build_loop()


    requests = [

        Request(
            prompt=f"request-{i}"
        )

        for i in range(batch_size)

    ]


    start = time.time()


    loop.submit_batch(
        requests
    )


    results = loop.run_once()


    end = time.time()


    latency = end - start


    print(
        "=" * 40
    )

    print(
        f"Batch size: {batch_size}"
    )


    print(
        f"Requests: {len(results)}"
    )


    print(
        f"Latency: {latency:.4f}s"
    )


    print(
        f"Throughput: "
        f"{len(results)/latency:.2f} req/s"
    )



if __name__ == "__main__":


    for batch_size in [
        1,
        2,
        4
    ]:

        benchmark(
            batch_size
        )