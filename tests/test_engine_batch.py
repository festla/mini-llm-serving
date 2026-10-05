import unittest

from services.mini_serving.engine.engine import (
    LLMEngine
)

from services.mini_serving.engine.request import (
    Request,
    RequestStatus
)


class FakeModelRunner:

    def generate(
        self,
        prompt,
        max_tokens
    ):

        return (
            f"generated:{prompt}"
        )


class FailingModelRunner:

    def generate(
        self,
        prompt,
        max_tokens
    ):

        if prompt == "B":

            raise RuntimeError(
                "fake inference error"
            )

        return (
            f"generated:{prompt}"
        )


class TestEngineBatch(unittest.TestCase):


    def test_generate_batch(self):

        engine = LLMEngine(
            runner=FakeModelRunner()
        )


        batch = [

            Request(prompt="A"),

            Request(prompt="B"),

            Request(prompt="C")

        ]


        results = engine.generate_batch(
            batch
        )


        self.assertEqual(
            len(results),
            3
        )


        for request in results:

            self.assertEqual(
                request.status,
                RequestStatus.FINISHED
            )

            self.assertIsNotNone(
                request.started_at
            )

            self.assertIsNotNone(
                request.finished_at
            )


        self.assertEqual(
            results[0].output_text,
            "generated:A"
        )

        self.assertEqual(
            results[1].output_text,
            "generated:B"
        )

        self.assertEqual(
            results[2].output_text,
            "generated:C"
        )


    def test_batch_failure_isolation(self):

        engine = LLMEngine(
            runner=FailingModelRunner()
        )


        batch = [

            Request(prompt="A"),

            Request(prompt="B"),

            Request(prompt="C")

        ]


        results = engine.generate_batch(
            batch
        )


        self.assertEqual(
            results[0].status,
            RequestStatus.FINISHED
        )

        self.assertEqual(
            results[1].status,
            RequestStatus.FAILED
        )

        self.assertEqual(
            results[2].status,
            RequestStatus.FINISHED
        )


        self.assertEqual(
            results[1].error,
            "fake inference error"
        )



if __name__ == "__main__":

    unittest.main()
