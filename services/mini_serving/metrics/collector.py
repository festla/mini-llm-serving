from threading import Lock

from services.mini_serving.engine.request import (
    RequestStatus
)


class MetricsCollector:

    def __init__(self):

        self._lock = Lock()

        self._total_requests = 0

        self._finished_requests = 0

        self._failed_requests = 0

        self._successful_requests = 0

        self._total_queue_time = 0.0

        self._total_inference_time = 0.0

        self._total_latency = 0.0

        self._current_batch_size = 0

        self._max_batch_size_observed = 0

        self._recorded_terminal_requests = set()


    def record_submit(
        self,
        request
    ):

        with self._lock:

            self._total_requests += 1


    def record_batch_start(
        self,
        batch
    ):

        batch_size = len(batch)

        with self._lock:

            self._current_batch_size = batch_size

            self._max_batch_size_observed = max(
                self._max_batch_size_observed,
                batch_size
            )


    def record_batch_end(self):

        with self._lock:

            self._current_batch_size = 0


    def record_terminal(
        self,
        request
    ):

        with self._lock:

            if (
                request.request_id
                in self._recorded_terminal_requests
            ):
                return

            if request.status == RequestStatus.FINISHED:

                self._finished_requests += 1

                self._successful_requests += 1

                if request.queue_time is not None:

                    self._total_queue_time += (
                        request.queue_time
                    )

                if request.inference_time is not None:

                    self._total_inference_time += (
                        request.inference_time
                    )

                if request.total_latency is not None:

                    self._total_latency += (
                        request.total_latency
                    )

            elif request.status == RequestStatus.FAILED:

                self._failed_requests += 1

            else:

                return

            self._recorded_terminal_requests.add(
                request.request_id
            )


    def snapshot(
        self,
        queue_size=0
    ):

        with self._lock:

            completed_requests = (
                self._finished_requests
                +
                self._failed_requests
            )

            in_flight_requests = max(
                0,
                self._total_requests
                -
                completed_requests
            )

            if self._successful_requests > 0:

                avg_queue_time_ms = (
                    self._total_queue_time
                    /
                    self._successful_requests
                    *
                    1000
                )

                avg_inference_time_ms = (
                    self._total_inference_time
                    /
                    self._successful_requests
                    *
                    1000
                )

                avg_total_latency_ms = (
                    self._total_latency
                    /
                    self._successful_requests
                    *
                    1000
                )

            else:

                avg_queue_time_ms = 0.0

                avg_inference_time_ms = 0.0

                avg_total_latency_ms = 0.0


            return {

                "requests": {

                    "total":
                        self._total_requests,

                    "finished":
                        self._finished_requests,

                    "failed":
                        self._failed_requests,

                    "in_flight":
                        in_flight_requests
                },

                "runtime": {

                    "queue_size":
                        queue_size,

                    "current_batch_size":
                        self._current_batch_size,

                    "max_batch_size_observed":
                        self._max_batch_size_observed
                },

                "performance": {

                    "avg_queue_time_ms":
                        round(
                            avg_queue_time_ms,
                            3
                        ),

                    "avg_inference_time_ms":
                        round(
                            avg_inference_time_ms,
                            3
                        ),

                    "avg_total_latency_ms":
                        round(
                            avg_total_latency_ms,
                            3
                        )
                }
            }
