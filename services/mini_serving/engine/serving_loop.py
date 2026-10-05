import time
import threading


class ServingLoop:


    def __init__(
        self,
        queue,
        scheduler,
        engine,
        metrics=None
    ):

        self.queue = queue
        self.scheduler = scheduler
        self.engine = engine
        self.metrics = metrics

        self.running = False

        self.thread = None



    def submit(
        self,
        request
    ):

        print(
            f"[Runtime] Submit request {request.request_id}"
        )

        self.queue.enqueue(
            request
        )

        if self.metrics is not None:

            self.metrics.record_submit(
                request
            )

        return request.request_id


    def submit_batch(
        self,
        requests
    ):

        request_ids = []


        for request in requests:

            request_id = self.submit(
                request
            )

            request_ids.append(
                request_id
            )


        return request_ids


    def run_once(self):

        batch = self.scheduler.schedule()

        if not batch:

            return []


        if self.metrics is not None:

            self.metrics.record_batch_start(
                batch
            )


        try:

            print(
                f"[Worker] Execute batch size={len(batch)}"
            )

            outputs = self.engine.generate_batch(
                batch
            )

            if self.metrics is not None:

                for request in outputs:

                    self.metrics.record_terminal(
                        request
                    )

            print(
                "[Worker] Batch finished"
            )

            return outputs


        finally:

            if self.metrics is not None:

                self.metrics.record_batch_end()

    def run_continuous_once(self):

        batch = self.scheduler.schedule()


        if not batch:

            return []


        print(
            f"[ContinuousWorker] Step batch size={len(batch)}"
        )

        print(
            "[ContinuousWorker] Active: "
            +
            ",".join(
                request.prompt
                for request in batch
            )
        )


        active_requests = self.engine.step(
            batch
        )


        finished_requests = [

            request

            for request in batch

            if request.status.value == "finished"
        ]


        if finished_requests:

            print(
                "[ContinuousWorker] Finished: "
                +
                ",".join(
                    request.prompt
                    for request in finished_requests
                )
            )


        return active_requests


    def start_continuous(self):

        self.running = True


        print(
            "Continuous ServingLoop started"
        )


        while self.running:

            batch = self.scheduler.schedule()


            if not batch:

                time.sleep(
                    0.01
                )

                continue


            self.run_continuous_once()


    def start(self):

        print(
            "ServingLoop started"
        )

        self.running = True


        while self.running:

            outputs = self.run_once()


            if not outputs:

                time.sleep(
                    0.01
                )



    def start_background(self):

        if self.thread is not None:

            return


        self.thread = threading.Thread(
            target=self.start,
            daemon=True
        )


        self.thread.start()



    def stop(self):

        print(
            "ServingLoop stopped"
        )

        self.running = False



    def join(self):

        if self.thread:

            self.thread.join()


    def wait_for_request(
        self,
        request,
        timeout=None
    ):

        finished = request.finished_event.wait(
            timeout
        )


        if not finished:

            raise TimeoutError(
                "request timeout"
            )


        return request