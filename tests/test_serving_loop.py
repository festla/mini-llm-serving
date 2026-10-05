import unittest


from services.mini_serving.engine.request import (
    Request,
    RequestStatus
)

from services.mini_serving.engine.request_queue import (
    RequestQueue
)

from services.mini_serving.engine.scheduler import (
    StaticBatchScheduler
)

from services.mini_serving.engine.engine import (
    LLMEngine
)

from services.mini_serving.engine.serving_loop import (
    ServingLoop
)



class FakeModelRunner:


    def __init__(self):

        self.calls = []



    def generate(
        self,
        prompt,
        max_tokens
    ):

        self.calls.append(
            prompt
        )

        return (
            f"generated:{prompt}"
        )



class TestServingLoop(unittest.TestCase):


    def create_loop(self):

        runner = FakeModelRunner()


        engine = LLMEngine(
            runner=runner
        )


        queue = RequestQueue()


        scheduler = StaticBatchScheduler(
            queue,
            max_batch_size=3
        )


        loop = ServingLoop(
            queue,
            scheduler,
            engine
        )


        return (
            loop,
            runner
        )



    def test_submit_request(self):

        loop, runner = self.create_loop()


        request = Request(
            prompt="A"
        )


        request_id = loop.submit(
            request
        )


        self.assertEqual(
            request_id,
            request.request_id
        )


        self.assertEqual(
            request.status,
            RequestStatus.WAITING
        )



        self.assertEqual(
            loop.queue.size(),
            1
        )



    def test_run_once(self):

        loop, runner = self.create_loop()


        requests = [

            Request(prompt="A"),
            Request(prompt="B"),
            Request(prompt="C"),
            Request(prompt="D")

        ]


        for request in requests:

            loop.submit(
                request
            )



        outputs = loop.run_once()



        self.assertEqual(
            len(outputs),
            3
        )


        self.assertEqual(
            loop.queue.size(),
            1
        )


        self.assertEqual(
            runner.calls,
            [
                "A",
                "B",
                "C"
            ]
        )


        for req in outputs:

            self.assertEqual(
                req.status,
                RequestStatus.FINISHED
            )


    def test_empty_run(self):

        loop, runner = self.create_loop()


        outputs = loop.run_once()


        self.assertEqual(
            outputs,
            []
        )


    def test_wait_for_request(self):

        request = Request(
            prompt="hello"
        )


        request.finished_event.set()

        loop, runner = self.create_loop()


        result = loop.wait_for_request(
            request
        )


        self.assertEqual(
            result,
            request
        )


if __name__ == "__main__":

    unittest.main()
