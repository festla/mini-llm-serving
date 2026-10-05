from services.mini_serving.memory.block_manager import (
    BlockManager
)

from services.mini_serving.memory.paged_decode import (
    PagedDecodeEngine
)



def main():


    manager = BlockManager(
        num_blocks=8,
        block_size=4
    )


    engine = PagedDecodeEngine(
        manager
    )


    engine.add_request(
        "A",
        6
    )


    engine.add_request(
        "B",
        10
    )


    for step in range(1,12):


        print(
            "\n================"
        )

        print(
            f"Decode Step {step}"
        )


        finished = engine.step()


        print(
            "Finished:",
            finished
        )


        print(
            engine.state()
        )


        if not engine.requests:

            break



if __name__ == "__main__":

    main()