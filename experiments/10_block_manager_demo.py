from services.mini_serving.memory.block_manager import (
    BlockManager
)



def main():


    manager = BlockManager(
        num_blocks=4,
        block_size=4
    )


    print(
        "Initial:"
    )

    print(
        manager.state()
    )


    print(
        "\nAppend tokens to A"
    )


    for token in [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F"
    ]:

        manager.append_token(
            "request_A",
            token
        )


    print(
        manager.state()
    )


    print(
        "\nRequest A blocks:"
    )

    print(
        manager.get_request_blocks(
            "request_A"
        )
    )


    print(
        "\nFree request"
    )


    manager.free_request(
        "request_A"
    )


    print(
        manager.state()
    )



if __name__ == "__main__":

    main()