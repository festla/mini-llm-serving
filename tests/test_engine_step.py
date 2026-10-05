import unittest


from services.mini_serving.engine.engine import (
    LLMEngine
)


from services.mini_serving.engine.request import (
    Request,
    RequestStatus
)



class TestEngineStep(unittest.TestCase):


    def test_single_step_generation(self):

        engine = LLMEngine()


        request = Request(
            prompt="hello",
            max_tokens=3
        )


        engine.step(
            [
                request
            ]
        )


        self.assertEqual(
            request.status,
            RequestStatus.DECODING
        )


        self.assertEqual(
            request.current_step,
            1
        )


        self.assertEqual(
            len(request.generated_tokens),
            1
        )


    def test_multiple_steps_finish(self):

        engine = LLMEngine()


        request = Request(
            prompt="hello",
            max_tokens=2
        )


        engine.step(
            [request]
        )


        self.assertFalse(
            request.is_finished
        )


        engine.step(
            [request]
        )


        self.assertTrue(
            request.is_finished
        )


        self.assertEqual(
            request.status,
            RequestStatus.FINISHED
        )


    def test_step_remove_finished_request(self):

        engine = LLMEngine()


        request_a = Request(
            prompt="A",
            max_tokens=1
        )


        request_b = Request(
            prompt="B",
            max_tokens=3
        )


        active = engine.step(
            [
                request_a,
                request_b
            ]
        )


        self.assertEqual(
            len(active),
            1
        )


        self.assertEqual(
            active[0],
            request_b
        )


        self.assertEqual(
            request_a.status,
            RequestStatus.FINISHED
        )


if __name__ == "__main__":

    unittest.main()