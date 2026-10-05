from services.mini_serving.memory.paged_block import (
    KVBlock
)



class BlockManager:


    def __init__(
        self,
        num_blocks,
        block_size
    ):

        self.blocks = [

            KVBlock(
                block_id=i,
                capacity=block_size
            )

            for i in range(num_blocks)

        ]


        self.request_blocks = {}



    def allocate_block(
        self,
        request_id
    ):


        for block in self.blocks:


            if block.request_id is None:


                block.request_id = request_id


                self.request_blocks.setdefault(
                    request_id,
                    []
                ).append(
                    block.block_id
                )


                return block.block_id



        raise RuntimeError(
            "No free blocks"
        )



    def append_token(
        self,
        request_id,
        token
    ):


        block_ids = self.request_blocks.get(
            request_id,
            []
        )


        if not block_ids:

            self.allocate_block(
                request_id
            )


            block_ids = self.request_blocks[
                request_id
            ]



        last_block = self.blocks[
            block_ids[-1]
        ]



        if last_block.is_full():


            self.allocate_block(
                request_id
            )


            last_block = self.blocks[
                self.request_blocks[request_id][-1]
            ]



        last_block.append_token(
            token
        )



    def get_request_blocks(
        self,
        request_id
    ):

        return self.request_blocks.get(
            request_id,
            []
        )



    def free_request(
        self,
        request_id
    ):


        block_ids = self.request_blocks.get(
            request_id,
            []
        )


        for block_id in block_ids:

            self.blocks[
                block_id
            ].reset()



        del self.request_blocks[
            request_id
        ]



    def state(self):

        return [

            (
                block.block_id,

                block.request_id,

                block.tokens

            )

            for block in self.blocks

        ]