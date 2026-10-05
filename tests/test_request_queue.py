import unittest

from services.mini_serving.engine.request import (
    Request
)

from services.mini_serving.engine.request_queue import (
    RequestQueue
)


class TestRequestQueue(unittest.TestCase):

    def test_empty_queue(self):

        queue = RequestQueue()

        self.assertTrue(
            queue.empty()
        )

        self.assertEqual(
            queue.size(),
            0
        )

        self.assertIsNone(
            queue.dequeue()
        )


    def test_enqueue(self):

        queue = RequestQueue()

        request = Request(
            prompt="hello"
        )

        queue.enqueue(
            request
        )

        self.assertFalse(
            queue.empty()
        )

        self.assertEqual(
            queue.size(),
            1
        )


    def test_fifo(self):

        queue = RequestQueue()

        request_a = Request(
            prompt="A"
        )

        request_b = Request(
            prompt="B"
        )

        request_c = Request(
            prompt="C"
        )


        queue.enqueue(
            request_a
        )

        queue.enqueue(
            request_b
        )

        queue.enqueue(
            request_c
        )


        self.assertIs(
            queue.dequeue(),
            request_a
        )

        self.assertIs(
            queue.dequeue(),
            request_b
        )

        self.assertIs(
            queue.dequeue(),
            request_c
        )


        self.assertTrue(
            queue.empty()
        )


if __name__ == "__main__":

    unittest.main()
