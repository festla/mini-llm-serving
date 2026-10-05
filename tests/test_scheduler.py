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


class TestStaticBatchScheduler(unittest.TestCase):


    def test_empty_queue(self):

        queue = RequestQueue()

        scheduler = StaticBatchScheduler(
            queue,
            max_batch_size=3
        )


        batch = scheduler.schedule()


        self.assertEqual(
            batch,
            []
        )


    def test_schedule_batch(self):

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

        request_d = Request(
            prompt="D"
        )

        request_e = Request(
            prompt="E"
        )


        for request in [
            request_a,
            request_b,
            request_c,
            request_d,
            request_e
        ]:

            queue.enqueue(
                request
            )


        scheduler = StaticBatchScheduler(
            queue,
            max_batch_size=3
        )


        batch1 = scheduler.schedule()


        self.assertEqual(
            len(batch1),
            3
        )

        self.assertIs(
            batch1[0],
            request_a
        )

        self.assertIs(
            batch1[1],
            request_b
        )

        self.assertIs(
            batch1[2],
            request_c
        )


        self.assertEqual(
            queue.size(),
            2
        )


        batch2 = scheduler.schedule()


        self.assertEqual(
            len(batch2),
            2
        )

        self.assertIs(
            batch2[0],
            request_d
        )

        self.assertIs(
            batch2[1],
            request_e
        )


        self.assertTrue(
            queue.empty()
        )


    def test_scheduler_does_not_change_status(self):

        queue = RequestQueue()

        request = Request(
            prompt="hello"
        )

        queue.enqueue(
            request
        )


        scheduler = StaticBatchScheduler(
            queue,
            max_batch_size=1
        )


        batch = scheduler.schedule()


        self.assertEqual(
            batch[0].status,
            RequestStatus.WAITING
        )


    def test_invalid_batch_size(self):

        queue = RequestQueue()


        with self.assertRaises(
            ValueError
        ):

            StaticBatchScheduler(
                queue,
                max_batch_size=0
            )



if __name__ == "__main__":

    unittest.main()
