class ContinuousAllocator:

    def __init__(self, total_tokens):
        self.total_tokens = total_tokens
        
        # memory layout
        # 0 表示空闲
        self.memory = [0] * total_tokens

        self.allocations = {}


    def allocate(self, request_id, size):

        # first fit
        start = None
        length = 0


        for i, value in enumerate(self.memory):

            if value == 0:
                length += 1

                if length == size:
                    start = i - size + 1
                    break

            else:
                length = 0


        if start is None:
            return False


        for i in range(start, start + size):
            self.memory[i] = request_id


        self.allocations[request_id] = (
            start,
            size
        )

        return True



    def free(self, request_id):

        if request_id not in self.allocations:
            return


        start, size = self.allocations[request_id]


        for i in range(start, start + size):
            self.memory[i] = 0


        del self.allocations[request_id]



    def show(self):

        print(
            "".join(
                "." if x == 0 else str(x)
                for x in self.memory
            )
        )


    def free_tokens(self):

        return self.memory.count(0)


    # ============================
    # Phase 5.2.2 新增
    # Fragmentation Metrics
    # ============================


    def largest_free_block(
        self
    ):

        largest = 0

        current = 0


        for value in self.memory:


            if value == 0:

                current += 1

                largest = max(
                    largest,
                    current
                )


            else:

                current = 0



        return largest




    def fragmentation_ratio(
        self
    ):

        total_free = self.free_tokens()


        if total_free == 0:

            return 0



        largest = self.largest_free_block()


        return (

            1
            -
            largest / total_free

        )

    
    def state(self):

        result = []

        for cell in self.memory:

            if cell == 0:
                result.append(".")

            else:
                result.append(str(cell))


        return "".join(result)