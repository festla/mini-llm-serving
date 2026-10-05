from services.mini_serving.engine.request_queue import (
    RequestQueue
)

from services.mini_serving.engine.request import (
    RequestStatus
)

class StaticBatchScheduler:

    def __init__(
        self,
        request_queue: RequestQueue,
        max_batch_size: int = 4
    ):

        if max_batch_size <= 0:

            raise ValueError(
                "max_batch_size must be greater than 0"
            )

        self.request_queue = request_queue

        self.max_batch_size = max_batch_size


    def schedule(self):

        batch = []


        while (
            len(batch) < self.max_batch_size
            and not self.request_queue.empty()
        ):

            request = (
                self.request_queue.dequeue()
            )


            if request is None:

                break


            batch.append(
                request
            )

        return batch


class ContinuousBatchScheduler:


    def __init__(
        self,
        request_queue,
        max_batch_size=4
    ):

        self.request_queue = request_queue

        self.max_batch_size = max_batch_size

        self.active_requests = []


    def schedule(self):

        # 1.保留未完成请求

        active = []

        for request in self.active_requests:

            if request.status not in (
                RequestStatus.FINISHED,
                RequestStatus.FAILED
            ):

                active.append(
                    request
                )

        self.active_requests = active


        # 2.补充新的请求

        while (
            len(self.active_requests)
            <
            self.max_batch_size
            and 
            not self.request_queue.empty()
        ):

            request = (
                self.request_queue.dequeue()
            )


            self.active_requests.append(
                request
            )


        return self.active_requests