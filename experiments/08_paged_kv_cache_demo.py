from services.mini_serving.memory.block_allocator import (
    BlockAllocator
)

from services.mini_serving.memory.block_table import (
    BlockTable
)

from services.mini_serving.memory.paged_kv_cache import (
    PagedKVCacheManager
)



def main():


    allocator = BlockAllocator(
        num_blocks=8
    )


    table = BlockTable()


    manager = PagedKVCacheManager(
        allocator,
        table,
        block_size=4
    )


    print(
        "Create Request A"
    )


    manager.create_request(
        "A"
    )


    print(
        manager.state()
    )


    print(
        "\nAppend 6 tokens"
    )


    manager.append_tokens(
        "A",
        6
    )


    print(
        manager.state()
    )


    print(
        "\nAppend 3 tokens"
    )


    manager.append_tokens(
        "A",
        3
    )


    print(
        manager.state()
    )


    print(
        "\nRelease A"
    )


    manager.release(
        "A"
    )


    print(
        manager.state()
    )



if __name__ == "__main__":

    main()