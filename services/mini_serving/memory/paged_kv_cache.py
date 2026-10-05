import math


class PagedKVCacheManager:


    def __init__(
        self,
        allocator,
        block_table,
        block_size=4
    ):

        self.allocator = allocator

        self.block_table = block_table

        self.block_size = block_size

        self.sequence_lengths = {}



    def create_request(
        self,
        request_id
    ):

        self.block_table.add_request(
            request_id
        )

        self.sequence_lengths[
            request_id
        ] = 0



    def append_tokens(
        self,
        request_id,
        num_tokens
    ):


        if request_id not in self.sequence_lengths:

            raise ValueError(
                "request not exists"
            )


        old_length = self.sequence_lengths[
            request_id
        ]


        new_length = (
            old_length
            +
            num_tokens
        )


        old_blocks = math.ceil(
            old_length
            /
            self.block_size
        )


        new_blocks = math.ceil(
            new_length
            /
            self.block_size
        )


        need_blocks = (
            new_blocks
            -
            old_blocks
        )


        if need_blocks > 0:


            blocks = self.allocator.allocate(
                request_id,
                need_blocks
            )


            if blocks is None:

                raise RuntimeError(
                    "No enough blocks"
                )


            for block_id in blocks:

                self.block_table.append_block(
                    request_id,
                    block_id
                )


        self.sequence_lengths[
            request_id
        ] = new_length



    def release(
        self,
        request_id
    ):


        blocks = self.block_table.get_blocks(
            request_id
        )


        self.allocator.free(
            blocks
        )


        self.block_table.remove_request(
            request_id
        )


        del self.sequence_lengths[
            request_id
        ]



    def state(self):

        return {

            "sequence_lengths":
                self.sequence_lengths,

            "block_table":
                self.block_table.state(),

            "blocks":
                self.allocator.state()

        }