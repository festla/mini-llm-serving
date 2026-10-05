class BlockTable:


    def __init__(self):

        self.tables = {}



    def add_request(
        self,
        request_id
    ):

        self.tables[
            request_id
        ] = []



    def append_block(
        self,
        request_id,
        physical_block_id
    ):

        if request_id not in self.tables:

            self.add_request(
                request_id
            )


        self.tables[
            request_id
        ].append(
            physical_block_id
        )



    def get_blocks(
        self,
        request_id
    ):

        return self.tables.get(
            request_id,
            []
        )



    def remove_request(
        self,
        request_id
    ):

        if request_id in self.tables:

            del self.tables[
                request_id
            ]



    def state(self):

        return self.tables