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


def print_request_state(
    name,
    request
):

    print(
        f"{name}: "
        f"status={request.status.value}, "
        f"step={request.current_step}/{request.max_tokens}"
    )


def main():

    queue = RequestQueue()


    scheduler = ContinuousBatchScheduler(
        queue,
        max_batch_size=2
    )


    # 当前 step() 不依赖真实模型 runner，
    # 因此 Demo 不加载 Qwen 权重。
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
        max_tokens=5
    )

    c = Request(
        prompt="C",
        max_tokens=2
    )

    d = Request(
        prompt="D",
        max_tokens=2
    )


    # 初始只到达 A、B
    loop.submit(a)
    loop.submit(b)


    all_requests = [
        a,
        b,
        c,
        d
    ]


    step = 0


    while True:

        step += 1


        print(
            "\n"
            + "=" * 50
        )

        print(
            f"Continuous Step {step}"
        )


        # 模拟运行过程中新的 HTTP 请求到达

        if step == 2:

            print(
                "[Arrival] Request C arrived"
            )

            loop.submit(
                c
            )


        if step == 4:

            print(
                "[Arrival] Request D arrived"
            )

            loop.submit(
                d
            )


        loop.run_continuous_once()


        print(
            "\nRequest states:"
        )


        print_request_state(
            "A",
            a
        )

        print_request_state(
            "B",
            b
        )

        print_request_state(
            "C",
            c
        )

        print_request_state(
            "D",
            d
        )


        all_finished = all(
            request.status
            ==
            RequestStatus.FINISHED

            for request in all_requests
        )


        if all_finished:

            break


    print(
        "\n"
        + "=" * 50
    )

    print(
        "All requests finished."
    )


if __name__ == "__main__":

    main()