from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
)


class RuntimePrometheusCollector:

    def __init__(
        self,
        metrics,
        queue_size_fn
    ):

        self.metrics = metrics

        self.queue_size_fn = queue_size_fn


    def collect(self):

        snapshot = self.metrics.snapshot(
            queue_size=self.queue_size_fn()
        )

        requests = snapshot["requests"]

        runtime = snapshot["runtime"]

        performance = snapshot["performance"]


        # =========================
        # Request Counters
        # =========================

        total_requests = CounterMetricFamily(
            "mini_serving_requests_total",
            "Total number of submitted requests"
        )

        total_requests.add_metric(
            [],
            requests["total"]
        )

        yield total_requests


        finished_requests = CounterMetricFamily(
            "mini_serving_requests_finished_total",
            "Total number of finished requests"
        )

        finished_requests.add_metric(
            [],
            requests["finished"]
        )

        yield finished_requests


        failed_requests = CounterMetricFamily(
            "mini_serving_requests_failed_total",
            "Total number of failed requests"
        )

        failed_requests.add_metric(
            [],
            requests["failed"]
        )

        yield failed_requests


        # =========================
        # Runtime Gauges
        # =========================

        in_flight_requests = GaugeMetricFamily(
            "mini_serving_requests_in_flight",
            "Current number of in-flight requests"
        )

        in_flight_requests.add_metric(
            [],
            requests["in_flight"]
        )

        yield in_flight_requests


        queue_size = GaugeMetricFamily(
            "mini_serving_queue_size",
            "Current number of requests waiting in the queue"
        )

        queue_size.add_metric(
            [],
            runtime["queue_size"]
        )

        yield queue_size


        current_batch_size = GaugeMetricFamily(
            "mini_serving_current_batch_size",
            "Current scheduler batch size"
        )

        current_batch_size.add_metric(
            [],
            runtime["current_batch_size"]
        )

        yield current_batch_size


        max_batch_size = GaugeMetricFamily(
            "mini_serving_max_batch_size_observed",
            "Maximum scheduler batch size observed"
        )

        max_batch_size.add_metric(
            [],
            runtime["max_batch_size_observed"]
        )

        yield max_batch_size


        # =========================
        # Performance Gauges
        # Prometheus uses seconds
        # =========================

        avg_queue_time = GaugeMetricFamily(
            "mini_serving_request_queue_time_average_seconds",
            "Average request queue time in seconds"
        )

        avg_queue_time.add_metric(
            [],
            performance["avg_queue_time_ms"]
            / 1000
        )

        yield avg_queue_time


        avg_inference_time = GaugeMetricFamily(
            "mini_serving_request_inference_time_average_seconds",
            "Average request inference time in seconds"
        )

        avg_inference_time.add_metric(
            [],
            performance["avg_inference_time_ms"]
            / 1000
        )

        yield avg_inference_time


        avg_latency = GaugeMetricFamily(
            "mini_serving_request_latency_average_seconds",
            "Average end-to-end request latency in seconds"
        )

        avg_latency.add_metric(
            [],
            performance["avg_total_latency_ms"]
            / 1000
        )

        yield avg_latency
