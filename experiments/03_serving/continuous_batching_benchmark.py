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


WORKLOAD = [
    ("A", 2),
    ("B", 6),
    ("C", 3),
    ("D", 2),
    ("E", 4),
]


MAX_BATCH_SIZE = 2


def build_requests():

    return [
        Request(
            prompt=name,
            max_tokens=max_tokens
        )
        for name, max_tokens in WORKLOAD
    ]


def build_step_engine():

    # 当前 step() 不依赖真实 ModelRunner，
    # 因此 Benchmark 不加载 Qwen。
    return LLMEngine.__new__(
        LLMEngine
    )


def calculate_metrics(
    total_rounds,
    useful_slots,
    completion_rounds
):

    capacity_slots = (
        total_rounds
        *
        MAX_BATCH_SIZE
    )


    utilization = (
        useful_slots
        /
        capacity_slots
    )


    normalized_throughput = (
        useful_slots
        /
        total_rounds
    )


    average_completion_round = (
        sum(
            completion_rounds.values()
        )
        /
        len(completion_rounds)
    )


    return {
        "total_rounds":
            total_rounds,

        "useful_slots":
            useful_slots,

        "capacity_slots":
            capacity_slots,

        "utilization":
            utilization,

        "normalized_throughput":
            normalized_throughput,

        "average_completion_round":
            average_completion_round,

        "completion_rounds":
            completion_rounds
    }


def benchmark_static():

    requests = build_requests()

    engine = build_step_engine()


    total_rounds = 0

    useful_slots = 0

    completion_rounds = {}


    print(
        "\n"
        + "=" * 60
    )

    print(
        "Static Batch"
    )


    for start in range(
        0,
        len(requests),
        MAX_BATCH_SIZE
    ):

        active = requests[
            start:
            start + MAX_BATCH_SIZE
        ]


        print(
            "\nNew static batch:",
            [
                request.prompt
                for request in active
            ]
        )


        while active:

            total_rounds += 1


            current_batch = list(
                active
            )


            useful_slots += len(
                current_batch
            )


            print(
                f"Round {total_rounds}: "
                f"{[r.prompt for r in current_batch]}"
            )


            active = engine.step(
                current_batch
            )


            for request in current_batch:

                if (
                    request.status
                    ==
                    RequestStatus.FINISHED
                    and
                    request.prompt
                    not in completion_rounds
                ):

                    completion_rounds[
                        request.prompt
                    ] = total_rounds


    return calculate_metrics(
        total_rounds,
        useful_slots,
        completion_rounds
    )


def benchmark_continuous():

    requests = build_requests()


    queue = RequestQueue()


    scheduler = ContinuousBatchScheduler(
        queue,
        max_batch_size=MAX_BATCH_SIZE
    )


    engine = build_step_engine()


    for request in requests:

        queue.enqueue(
            request
        )


    total_rounds = 0

    useful_slots = 0

    completion_rounds = {}


    print(
        "\n"
        + "=" * 60
    )

    print(
        "Continuous Batch"
    )


    while (
        len(completion_rounds)
        <
        len(requests)
    ):

        batch = scheduler.schedule()


        if not batch:

            break


        total_rounds += 1


        current_batch = list(
            batch
        )


        useful_slots += len(
            current_batch
        )


        print(
            f"Round {total_rounds}: "
            f"{[r.prompt for r in current_batch]}"
        )


        engine.step(
            current_batch
        )


        for request in current_batch:

            if (
                request.status
                ==
                RequestStatus.FINISHED
                and
                request.prompt
                not in completion_rounds
            ):

                completion_rounds[
                    request.prompt
                ] = total_rounds


    return calculate_metrics(
        total_rounds,
        useful_slots,
        completion_rounds
    )


def print_metrics(
    name,
    metrics
):

    print(
        "\n"
        + "-" * 60
    )

    print(
        name
    )

    print(
        f"Total decode rounds: "
        f"{metrics['total_rounds']}"
    )

    print(
        f"Useful token slots: "
        f"{metrics['useful_slots']}"
    )

    print(
        f"Available token slots: "
        f"{metrics['capacity_slots']}"
    )

    print(
        f"Batch utilization: "
        f"{metrics['utilization'] * 100:.2f}%"
    )

    print(
        f"Normalized throughput: "
        f"{metrics['normalized_throughput']:.2f} "
        f"token-slots/round"
    )

    print(
        f"Average completion round: "
        f"{metrics['average_completion_round']:.2f}"
    )

    print(
        f"Completion rounds: "
        f"{metrics['completion_rounds']}"
    )


def main():

    print(
        "Workload:",
        WORKLOAD
    )

    print(
        "Max batch size:",
        MAX_BATCH_SIZE
    )


    static_metrics = (
        benchmark_static()
    )


    continuous_metrics = (
        benchmark_continuous()
    )


    print_metrics(
        "Static Batch Result",
        static_metrics
    )


    print_metrics(
        "Continuous Batch Result",
        continuous_metrics
    )


    round_reduction = (
        (
            static_metrics["total_rounds"]
            -
            continuous_metrics["total_rounds"]
        )
        /
        static_metrics["total_rounds"]
    )


    throughput_gain = (
        (
            continuous_metrics[
                "normalized_throughput"
            ]
            /
            static_metrics[
                "normalized_throughput"
            ]
        )
        -
        1
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "Comparison"
    )

    print(
        f"Decode round reduction: "
        f"{round_reduction * 100:.2f}%"
    )

    print(
        f"Normalized throughput gain: "
        f"{throughput_gain * 100:.2f}%"
    )


if __name__ == "__main__":

    main()