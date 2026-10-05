import time

from services.mini_serving.engine.request import Request
from services.mini_serving.engine.request_queue import RequestQueue
from services.mini_serving.engine.scheduler import StaticBatchScheduler
from services.mini_serving.engine.engine import LLMEngine
from services.mini_serving.engine.serving_loop import ServingLoop


queue = RequestQueue()


scheduler = StaticBatchScheduler(
    queue,
    max_batch_size=2
)


engine = LLMEngine()


loop = ServingLoop(
    queue,
    scheduler,
    engine
)


loop.start_background()


time.sleep(1)


loop.submit(
    Request(
        prompt="解释KV Cache"
    )
)


time.sleep(5)


loop.stop()