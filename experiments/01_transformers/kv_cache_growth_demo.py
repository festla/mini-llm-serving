# 模拟 LLM Serving 中 KV Cache 动态增长


BLOCK_SIZE = 4

TOTAL_BLOCKS = 20



# =====================================================
# 1. Continuous KV Cache
# =====================================================

print("===== Continuous KV Cache =====")


class ContinuousKVCache:


    def __init__(self):

        self.capacity = 4

        self.length = 0

        self.copy_times = 0



    def append_token(self):

        self.length += 1


        # 空间不足，需要扩容

        if self.length > self.capacity:

            old_capacity = self.capacity


            self.capacity *= 2


            # 模拟数据搬迁

            self.copy_times += old_capacity


            print(
                f"Resize:"
                f"{old_capacity} -> {self.capacity}"
            )



    def info(self):

        print(
            "length:",
            self.length,
            "capacity:",
            self.capacity,
            "copy:",
            self.copy_times
        )



cache_A = ContinuousKVCache()



for i in range(20):

    cache_A.append_token()


print("\nFinal:")
cache_A.info()





# =====================================================
# 2. Paged KV Cache
# =====================================================


print("\n===== Paged KV Cache =====")



class PagedKVCache:



    def __init__(self):

        self.blocks = []

        self.length = 0



    def append_token(self):


        self.length += 1


        # 当前block不足

        if (
            len(self.blocks) == 0
            or self.length > len(self.blocks)*BLOCK_SIZE
        ):

            new_block = len(self.blocks)

            self.blocks.append(new_block)


            print(
                "Allocate block:",
                new_block
            )



    def info(self):

        print(
            "length:",
            self.length,
            "blocks:",
            self.blocks
        )





cache_B = PagedKVCache()



for i in range(20):

    cache_B.append_token()



print("\nFinal:")
cache_B.info()
