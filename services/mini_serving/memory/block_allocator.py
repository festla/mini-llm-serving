from dataclasses import dataclass



@dataclass
class Block:

    block_id: int

    allocated: bool = False

    owner: str | None = None


class BlockAllocator:


    def __init__(
        self,
        num_blocks
    ):

        self.blocks = [

            Block(
                block_id=i
            )

            for i in range(
                num_blocks
            )

        ]



    def allocate(
        self,
        owner,
        num_blocks
    ):


        free_blocks = [

            block

            for block in self.blocks

            if not block.allocated

        ]


        if len(free_blocks) < num_blocks:

            return None



        allocated = free_blocks[
            :num_blocks
        ]


        for block in allocated:

            block.allocated = True

            block.owner = owner



        return [

            block.block_id

            for block in allocated

        ]



    def free(
        self,
        block_ids
    ):


        for block_id in block_ids:

            block = self.blocks[
                block_id
            ]


            block.allocated = False

            block.owner = None



    def state(self):

        return [

            (
                block.block_id,

                block.owner

            )

            for block in self.blocks

        ]
