import unittest

from services.mini_serving.engine.request import (
    Request,
    RequestStatus
)


class TestRequestLifecycle(unittest.TestCase):


    def test_normal_lifecycle(self):

        request = Request(
            prompt="Explain KV Cache",
            max_tokens=32
        )


        self.assertEqual(
            request.status,
            RequestStatus.WAITING
        )

        self.assertIsNone(
            request.started_at
        )

        self.assertIsNone(
            request.finished_at
        )


        request.mark_running()


        self.assertEqual(
            request.status,
            RequestStatus.RUNNING
        )

        self.assertIsNotNone(
            request.started_at
        )


        request.mark_finished(
            "KV Cache stores..."
        )


        self.assertEqual(
            request.status,
            RequestStatus.FINISHED
        )

        self.assertEqual(
            request.output_text,
            "KV Cache stores..."
        )

        self.assertIsNotNone(
            request.finished_at
        )


    def test_empty_prompt(self):

        with self.assertRaises(
            ValueError
        ):

            Request(
                prompt="",
                max_tokens=32
            )


    def test_invalid_max_tokens(self):

        with self.assertRaises(
            ValueError
        ):

            Request(
                prompt="hello",
                max_tokens=0
            )


    def test_invalid_transition(self):

        request = Request(
            prompt="hello"
        )

        request.mark_running()

        request.mark_finished(
            "done"
        )


        with self.assertRaises(
            ValueError
        ):

            request.mark_running()


    def test_timing_metrics(self):

        request = Request(
            prompt="Explain KV Cache",
            max_tokens=32
        )


        self.assertIsNone(
            request.queue_time
        )

        self.assertIsNone(
            request.inference_time
        )

        self.assertIsNone(
            request.total_latency
        )


        request.mark_running()

        request.mark_finished(
            "done"
        )


        self.assertGreaterEqual(
            request.queue_time,
            0
        )

        self.assertGreaterEqual(
            request.inference_time,
            0
        )

        self.assertGreaterEqual(
            request.total_latency,
            0
        )


        self.assertAlmostEqual(
            request.total_latency,
            request.queue_time
            +
            request.inference_time,
            places=6
        )


    def test_token_lifecycle(self):

        request = Request(
            prompt="hello",
            max_tokens=3
        )


        request.mark_prefill()

        self.assertEqual(
            request.status,
            RequestStatus.PREFILL
        )


        request.mark_decoding()


        request.append_token(
            "a"
        )

        request.append_token(
            "b"
        )


        self.assertEqual(
            request.current_step,
            2
        )


        self.assertFalse(
            request.is_finished
        )


        request.append_token(
            "c"
        )


        self.assertTrue(
            request.is_finished
        )

if __name__ == "__main__":

    unittest.main()
