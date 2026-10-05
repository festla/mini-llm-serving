class PagedDecodeEngine:


    def __init__(
        self,
        block_manager
    ):

        self.block_manager = block_manager


        self.requests = {}


        self.finished = {}



    def add_request(
        self,
        request_id,
        max_tokens
    ):


        self.requests[
            request_id
        ] = {

            "generated": 0,

            "max_tokens": max_tokens

        }



    def step(self):

        finished = []


        for request_id in list(
            self.requests.keys()
        ):


            state = self.requests[
                request_id
            ]


            # generate one token

            token_id = (
                state["generated"]
            )


            self.block_manager.append_token(
                request_id,
                token_id
            )


            state["generated"] += 1



            if (
                state["generated"]
                >=
                state["max_tokens"]
            ):


                self.block_manager.free_request(
                    request_id
                )


                finished.append(
                    request_id
                )

                del self.requests[
                    request_id
                ]



        return finished



    def state(self):

        return {

            "active":
                self.requests,

            "blocks":
                self.block_manager.state()

        }