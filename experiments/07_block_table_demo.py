from services.mini_serving.memory.block_table import (
    BlockTable
)



def main():

    table = BlockTable()


    print(
        "Initial:"
    )

    print(
        table.state()
    )


    print(
        "\nCreate request A"
    )


    table.add_request(
        "A"
    )


    print(
        table.state()
    )


    print(
        "\nAppend physical blocks"
    )


    table.append_block(
        "A",
        7
    )


    table.append_block(
        "A",
        2
    )


    table.append_block(
        "A",
        10
    )


    print(
        table.state()
    )


    print(
        "\nRequest A blocks:"
    )


    print(
        table.get_blocks(
            "A"
        )
    )



    print(
        "\nRemove request A"
    )


    table.remove_request(
        "A"
    )


    print(
        table.state()
    )



if __name__ == "__main__":

    main()