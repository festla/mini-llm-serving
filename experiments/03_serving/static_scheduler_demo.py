from services.mini_serving.engine.engine import (
    LLMEngine
)

from services.mini_serving.engine.request import (
    Request
)

from services.mini_serving.engine.serving_loop import (
    ServingLoop
)


class FakeModelRunner:

    def generate(
        self,
        prompt,
        max_tokens
    ):

        return f"generated:{prompt}"


engine = LLMEngine(
    runner=FakeModelRunner()
)


loop = ServingLoop(
    engine,
    max_batch_size=3
)


for name in [
    "A",
    "B",
    "C",
    "D",
    "E"
]:

    loop.submit(
        Request(
            prompt=name
        )
    )


loop.drain()


stats = loop.stats()


print(
    "Batch sizes:",
    stats["batch_sizes"]
)

print(
    "Total batches:",
    stats["total_batches"]
)

print(
    "Total requests:",
    stats["total_requests"]
)

print(
    "Average batch size:",
    stats["average_batch_size"]
)

print(
    "Pending requests:",
    stats["pending_requests"]
)
