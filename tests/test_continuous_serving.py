import unittest


from services.mini_serving.engine.request import (
    Request,
    RequestStatus
)

from services.mini_serving.engine.request_queue import (
    RequestQueue
)

from services.mini_serving.engine.scheduler import (
    ContinuousBatchScheduler
)

from services.mini_serving.engine.engine import (
    LLMEngine
)

from services.mini_serving.engine.serving_loop import (
    ServingLoop
)



class TestContinuousServing(unittest.TestCase):


    def test_dynamic_batch_replacement(self):

        queue = RequestQueue()


        scheduler = ContinuousBatchScheduler(
            queue,
            max_batch_size=2
        )


        engine = LLMEngine.__new__(
            LLMEngine
        )


        loop = ServingLoop(
            queue,
            scheduler,
            engine
        )


        a = Request(
            prompt="A",
            max_tokens=2
        )

        b = Request(
            prompt="B",
            max_tokens=4
        )

        c = Request(
            prompt="C",
            max_tokens=3
        )


        loop.submit(a)
        loop.submit(b)
        loop.submit(c)


        # Step 1
        loop.run_continuous_once()

        self.assertEqual(
            a.current_step,
            1
        )

        self.assertEqual(
            b.current_step,
            1
        )

        self.assertEqual(
            c.current_step,
            0
        )


        # Step 2:
        # A reaches max_tokens and finishes
        loop.run_continuous_once()

        self.assertEqual(
            a.status,
            RequestStatus.FINISHED
        )

        self.assertEqual(
            b.current_step,
            2
        )


        # Step 3:
        # Scheduler removes A and fills C
        loop.run_continuous_once()

        self.assertEqual(
            b.current_step,
            3
        )

        self.assertEqual(
            c.current_step,
            1
        )


        self.assertEqual(
            len(
                scheduler.active_requests
            ),
            2
        )


        self.assertIn(
            b,
            scheduler.active_requests
        )

        self.assertIn(
            c,
            scheduler.active_requests
        )



if __name__ == "__main__":

    unittest.main()