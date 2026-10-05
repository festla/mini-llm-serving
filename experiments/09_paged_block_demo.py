from services.mini_serving.memory.paged_block import (
    KVBlock
)



def main():


    block = KVBlock(
        block_id=0,
        capacity=4
    )


    print(
        "Initial:"
    )

    print(
        block
    )


    print(
        "\nAppend tokens"
    )


    for token in [
        "A",
        "B",
        "C"
    ]:

        block.append_token(
            token
        )


    print(
        block
    )


    print(
        "\nSize:"
    )

    print(
        block.size()
    )


    print(
        "\nFull:"
    )

    print(
        block.is_full()
    )



if __name__ == "__main__":

    main()