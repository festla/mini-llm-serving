from services.mini_serving.memory.kv_cache import (
    KVCache
)



def main():

    cache = KVCache(
        num_layers=2
    )


    print(
        "Initial seq len:",
        cache.memory_tokens()
    )


    cache.append(
        "K1",
        "V1"
    )


    print(
        "\nAfter token 1:"
    )

    print(
        "seq len:",
        cache.memory_tokens()
    )


    print(
        cache.get(0)
    )


    cache.append(
        "K2",
        "V2"
    )


    print(
        "\nAfter token 2:"
    )

    print(
        "seq len:",
        cache.memory_tokens()
    )


    print(
        cache.get(0)
    )



if __name__ == "__main__":

    main()