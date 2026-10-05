class KVMemoryEstimator:


    DTYPE_BYTES = {

        "fp32": 4,

        "fp16": 2,

        "bf16": 2

    }


    def __init__(
        self,
        num_layers,
        num_kv_heads,
        head_dim,
        dtype="fp16"
    ):

        self.num_layers = num_layers

        self.num_kv_heads = num_kv_heads

        self.head_dim = head_dim


        if dtype not in self.DTYPE_BYTES:

            raise ValueError(
                f"Unsupported dtype: {dtype}"
            )


        self.dtype = dtype

        self.bytes_per_element = (
            self.DTYPE_BYTES[dtype]
        )



    def bytes_per_token(self):


        return (

            2
            *
            self.num_layers
            *
            self.num_kv_heads
            *
            self.head_dim
            *
            self.bytes_per_element

        )



    def estimate(
        self,
        seq_len,
        num_requests=1
    ):


        total_bytes = (

            self.bytes_per_token()

            *

            seq_len

            *

            num_requests

        )


        return total_bytes



    @staticmethod
    def format_bytes(
        size
    ):

        units = [

            "B",

            "KB",

            "MB",

            "GB"

        ]


        value = float(size)


        for unit in units:

            if value < 1024:

                return (
                    f"{value:.2f} {unit}"
                )


            value /= 1024


        return (
            f"{value:.2f} TB"
        )