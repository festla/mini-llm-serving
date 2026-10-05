from services.mini_serving.memory.block_allocator import (
    BlockAllocator
)



def main():


    allocator = BlockAllocator(
        num_blocks=8
    )


    print(
        "Initial:"
    )

    print(
        allocator.state()
    )


    print(
        "\nAllocate A 3 blocks"
    )


    a_blocks = allocator.allocate(
        "A",
        3
    )


    print(
        "A blocks:",
        a_blocks
    )


    print(
        allocator.state()
    )



    print(
        "\nAllocate B 2 blocks"
    )


    b_blocks = allocator.allocate(
        "B",
        2
    )


    print(
        "B blocks:",
        b_blocks
    )


    print(
        allocator.state()
    )



    print(
        "\nFree A"
    )


    allocator.free(
        a_blocks
    )


    print(
        allocator.state()
    )



    print(
        "\nAllocate C 4 blocks"
    )


    c_blocks = allocator.allocate(
        "C",
        4
    )


    print(
        "C blocks:",
        c_blocks
    )


    print(
        allocator.state()
    )



if __name__ == "__main__":

    main()