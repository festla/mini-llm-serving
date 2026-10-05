from services.mini_serving.engine.request import (
    Request
)

from services.mini_serving.metrics.collector import (
    MetricsCollector
)


def test_metrics_collector():

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

    snapshot = metrics.snapshot(
        queue_size=0
    )

    assert (
        snapshot["requests"]["total"]
        == 1
    )

    assert (
        snapshot["requests"]["finished"]
        == 1
    )

    assert (
        snapshot["requests"]["failed"]
        == 0
    )

    assert (
        snapshot["requests"]["in_flight"]
        == 0
    )

    assert (
        snapshot["runtime"]["queue_size"]
        == 0
    )

    assert (
        snapshot["performance"]
        ["avg_total_latency_ms"]
        >= 0
    )


def test_terminal_metrics_are_idempotent():

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

    metrics.record_terminal(
        request
    )

    snapshot = metrics.snapshot()

    assert (
        snapshot["requests"]["finished"]
        == 1
    )
