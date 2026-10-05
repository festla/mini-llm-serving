class FakeEngine:


    def generate(self, requests):

        outputs=[]


        for req in requests:

            outputs.append(
                f"response for {req.prompt}"
            )

        return outputs
