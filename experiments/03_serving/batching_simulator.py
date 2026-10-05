
from dataclasses import dataclass
from collections import deque

@dataclass
class Request:

    name: str

    output_tokens: int

    generated: int = 0


    def finished(self):

        return self.generated >= self.output_tokens


# -----------------------
# Static Batch
# -----------------------

def static_batch(
        requests,
        max_batch_size=3
):

    print("\n===== Static Batch =====")

    total_utilization = 0
    total_steps = 0

    waiting = deque(requests)

    global_step = 0


    while waiting:

        # 静态 Batch：
        # 一次取固定的一批请求
        batch = []

        while (
            waiting
            and len(batch) < max_batch_size
        ):

            batch.append(
                waiting.popleft()
            )


        # 这一批必须全部完成，
        # 下一批才能进入
        while batch:

            global_step += 1


            print(
                f"Step {global_step}:",
                [r.name for r in batch]
            )

            total_steps += 1

            total_utilization += (
                len(batch)
                /
                max_batch_size
            )


            finished = []


            for r in batch:

                r.generated += 1

                if r.finished():

                    finished.append(r)


            for r in finished:

                batch.remove(r)

    print(
        "Total steps:",
        total_steps
    )

    print(
        "Average utilization:",
        f"{total_utilization / total_steps:.2f}"
    )


# ------------------
# continuous Batch
# ------------------

def continuous_batch(
        requests,
        max_batch_size=3
):
    print("\n===== Continuous Batch =====")

    total_utilization = 0
    total_steps = 0

    waiting = deque(requests)

    running = []

    step = 0

    while waiting or running:

        step += 1

        # 动态补充请求

        while (
            waiting
            and len(running) < max_batch_size
        ):

            req = waiting.popleft()

            running.append(req)

        print(
            f"Step {step}:",
            [r.name for r in running]
        )

        total_steps += 1
        total_utilization += (
            len(running)
            /
            max_batch_size
        )

        finished = []

        for r in running:
            r.generated += 1

            if r.finished():
                finished.append(r)


        for r in finished:
            running.remove(r)

    print(
        "Total steps:",
        total_steps
    )

    print(
        "Average utilization:",
        f"{total_utilization / total_steps:.2f}"
    )

if __name__ == "__main__":

    requests = [

        Request("A",8),
        Request("B",3),
        Request("C",5),
        Request("D",10),
        Request("E",4),
    ]

    static_batch(
        [
            Request("A",8),
            Request("B",3),
            Request("C",5),
            Request("D",10),
            Request("E",4),
        ],
        max_batch_size=3
    )

    continuous_batch(
        [
            Request("A", 8),
            Request("B", 3),
            Request("C", 5),
            Request("D", 10),
            Request("E", 4),
        ],
        max_batch_size=3
    )
