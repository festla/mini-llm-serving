from dataclasses import dataclass, field



@dataclass
class KVBlock:


    block_id: int


    capacity: int


    request_id: str | None = None


    tokens: list = field(
        default_factory=list
    )



    def is_full(self):

        return len(self.tokens) >= self.capacity



    def append_token(
        self,
        token
    ):


        if self.is_full():

            raise RuntimeError(
                "Block is full"
            )


        self.tokens.append(
            token
        )



    def reset(self):

        self.request_id = None

        self.tokens.clear()



    def size(self):

        return len(
            self.tokens
        )