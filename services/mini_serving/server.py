from fastapi import (
    FastAPI,
    HTTPException,
    Response
)

from prometheus_client import (
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
    generate_latest
)

from contextlib import asynccontextmanager

from pydantic import BaseModel


from services.mini_serving.engine.engine import (
    LLMEngine
)

from services.mini_serving.engine.request import (
    Request
)

from services.mini_serving.engine.request_queue import (
    RequestQueue
)

from services.mini_serving.engine.scheduler import (
    StaticBatchScheduler
)

from services.mini_serving.engine.serving_loop import (
    ServingLoop
)

from services.mini_serving.metrics.collector import (
    MetricsCollector
)

from services.mini_serving.metrics.prometheus_exporter import (
    RuntimePrometheusCollector
)

# =========================
# Initialize Serving Runtime
# =========================


queue = RequestQueue()


scheduler = StaticBatchScheduler(
    queue,
    max_batch_size=4
)


engine = LLMEngine()

metrics = MetricsCollector()

prometheus_registry = CollectorRegistry()

prometheus_registry.register(
    RuntimePrometheusCollector(
        metrics=metrics,
        queue_size_fn=queue.size
    )
)

serving_loop = ServingLoop(
    queue,
    scheduler,
    engine,
    metrics=metrics
)



@asynccontextmanager
async def lifespan(app: FastAPI):

    print(
        "Starting Serving Runtime..."
    )

    serving_loop.start_background()

    yield

    print(
        "Stopping Serving Runtime..."
    )

    serving_loop.stop()

app = FastAPI(
    title="Mini LLM Serving",
    lifespan=lifespan
)


class GenerateRequest(BaseModel):

    prompt: str

    max_tokens: int = 50

class BatchGenerateRequest(BaseModel):

    requests: list[GenerateRequest]


@app.get("/health")
def health():

    return {
        "status": "ok"
    }



@app.post("/generate")
def generate(
    body: GenerateRequest
):

    try:

        request = Request(
            prompt=body.prompt,
            max_tokens=body.max_tokens
        )


        # submit request
        serving_loop.submit(
            request
        )


        serving_loop.wait_for_request(
            request
        )


        return {

            "request_id":
                request.request_id,


            "status":
                request.status.value,


            "text":
                request.output_text,


            "time": {

                "queue_time":
                    request.queue_time,


                "inference_time":
                    request.inference_time,


                "total_latency":
                    request.total_latency
            }
        }


    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


@app.post("/generate_batch")
def generate_batch(
    body: BatchGenerateRequest
):

    try:

        requests = []

        for item in body.requests:

            request = Request(
                prompt=item.prompt,
                max_tokens=item.max_tokens
            )

            requests.append(
                request
            )

        serving_loop.submit_batch(
            requests
        )

        for request in requests:

            serving_loop.wait_for_request(
                request
            )

        return {

            "requests":[

                {
                    "request_id":
                        request.request_id,

                    "status":
                        request.status.value,

                    "text":
                        request.output_text
                }
            ]
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )
    
    except ValueError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )



@app.get("/metrics")
def get_metrics():

    return metrics.snapshot(
        queue_size=queue.size()
    )


@app.get("/metrics/prometheus")
def get_prometheus_metrics():

    content = generate_latest(
        prometheus_registry
    )

    return Response(
        content=content,
        media_type=CONTENT_TYPE_LATEST
    )