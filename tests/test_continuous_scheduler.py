import unittest


from services.mini_serving.engine.scheduler import (
    ContinuousBatchScheduler
)

from services.mini_serving.engine.request_queue import (
    RequestQueue
)

from services.mini_serving.engine.request import (
    Request,
    RequestStatus
)


class TestContinuousScheduler(unittest.TestCase):


    def test_initial_schedule(self):

        queue = RequestQueue()


        scheduler = ContinuousBatchScheduler(
            queue,
            max_batch_size=2
        )


        queue.enqueue(
            Request("A")
        )

        queue.enqueue(
            Request("B")
        )

        queue.enqueue(
            Request("C")
        )


        batch = scheduler.schedule()


        self.assertEqual(
            len(batch),
            2
        )



    def test_remove_finished_and_fill(self):


        queue = RequestQueue()


        scheduler = ContinuousBatchScheduler(
            queue,
            max_batch_size=2
        )


        a = Request("A")

        b = Request("B")

        c = Request("C")


        queue.enqueue(a)
        queue.enqueue(b)
        queue.enqueue(c)


        batch = scheduler.schedule()


        self.assertEqual(
            batch,
            [a, b]
        )

        # 模拟A完成

        a.status = RequestStatus.FINISHED


        batch = scheduler.schedule()


        self.assertEqual(
            len(batch),
            2
        )


        self.assertEqual(
            batch,
            [b, c]
        )


if __name__ == "__main__":
    unittest.main()