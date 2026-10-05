class KVCache:


    def __init__(
        self,
        num_layers
    ):

        self.num_layers = num_layers


        # 每层保存 token 对应的 KV
        self.layers = {

            layer_id:
            {
                "keys": [],
                "values": []
            }

            for layer_id
            in range(num_layers)

        }


        self.seq_len = 0



    def append(
        self,
        key,
        value
    ):

        """
        模拟一个 token 的 KV 写入
        """

        for layer_id in range(
            self.num_layers
        ):

            self.layers[layer_id]["keys"].append(
                key
            )

            self.layers[layer_id]["values"].append(
                value
            )


        self.seq_len += 1



    def get(
        self,
        layer_id
    ):

        return {

            "keys":
                self.layers[layer_id]["keys"],

            "values":
                self.layers[layer_id]["values"]

        }



    def memory_tokens(self):

        return self.seq_len