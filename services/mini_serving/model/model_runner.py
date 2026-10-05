from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

import torch

class ModelRunner:

    def __init__(
        self,
        model_name
    ):

        self.model_name = model_name

        print(
            "Loading Model..."
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map="auto"
            )
        )

        self.model.eval()

        print(
            "Model loaded."
        )

    def generate(
        self,
        prompt,
        max_tokens=50
    ):

        inputs = (
            self.tokenizer(
                prompt,
                return_tensors="pt"
            )
            .to(
                self.model.device
            )
        )

        with torch.no_grad():

            outputs = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens
                )
            )

        text = (
            self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
        )

        return text
