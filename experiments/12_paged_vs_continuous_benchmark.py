from services.mini_serving.memory.continuous_allocator import (
    ContinuousAllocator
)

from services.mini_serving.memory.block_manager import (
    BlockManager
)



def continuous_test():

    print(
        "\n========== Continuous =========="
    )


    allocator = ContinuousAllocator(
        32
    )


    a = allocator.allocate(
        "A",
        8
    )

    b = allocator.allocate(
        "B",
        8
    )

    c = allocator.allocate(
        "C",
        8
    )


    print(
        "After allocate A/B/C:"
    )

    print(
        allocator.state()
    )


    allocator.free(
        "B"
    )


    print(
        "\nAfter free B:"
    )

    print(
        allocator.state()
    )


    result = allocator.allocate(
        "D",
        10
    )


    print(
        "\nAllocate D(10):",
        result
    )



def paged_test():


    print(
        "\n========== Paged =========="
    )


    manager = BlockManager(
        num_blocks=8,
        block_size=4
    )


    for token in range(8):

        manager.append_token(
            "A",
            token
        )


    for token in range(8):

        manager.append_token(
            "B",
            token
        )


    for token in range(8):

        manager.append_token(
            "C",
            token
        )


    print(
        "After allocate A/B/C:"
    )

    print(
        manager.state()
    )


    manager.free_request(
        "B"
    )


    print(
        "\nAfter free B:"
    )

    print(
        manager.state()
    )


    for token in range(10):

        manager.append_token(
            "D",
            token
        )


    print(
        "\nAllocate D(10): success"
    )


    print(
        manager.state()
    )



if __name__ == "__main__":

    continuous_test()

    paged_test()