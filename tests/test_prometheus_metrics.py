from prometheus_client import (
    CollectorRegistry,
    generate_latest,
)

from services.mini_serving.engine.request import (
    Request
)

from services.mini_serving.metrics.collector import (
    MetricsCollector
)

from services.mini_serving.metrics.prometheus_exporter import (
    RuntimePrometheusCollector
)


def test_prometheus_metrics_export():

    metrics = MetricsCollector()

    request = Request(
        prompt="hello",
        max_tokens=1
    )

    metrics.record_submit(
        request
    )

    request.mark_running()

    request.mark_finished(
        "world"
    )

    metrics.record_terminal(
        request
    )


    registry = CollectorRegistry()

    registry.register(
        RuntimePrometheusCollector(
            metrics=metrics,
            queue_size_fn=lambda: 2
        )
    )


    output = generate_latest(
        registry
    ).decode("utf-8")


    assert (
        "mini_serving_requests_total 1.0"
        in output
    )

    assert (
        "mini_serving_requests_finished_total 1.0"
        in output
    )

    assert (
        "mini_serving_requests_failed_total 0.0"
        in output
    )

    assert (
        "mini_serving_requests_in_flight 0.0"
        in output
    )

    assert (
        "mini_serving_queue_size 2.0"
        in output
    )
