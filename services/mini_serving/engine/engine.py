from services.mini_serving.model.model_runner import (
    ModelRunner
)

from services.mini_serving.engine.request import (
    Request,
    RequestStatus
)


class LLMEngine:

    def __init__(
        self,
        runner=None
    ):

        if runner is None:

            runner = ModelRunner(
                "Qwen/Qwen2.5-0.5B-Instruct"
            )

        self.runner = runner


    def generate(
        self,
        request: Request
    ):

        try:

            request.mark_running()

            output_text = self.runner.generate(
                request.prompt,
                request.max_tokens
            )

            request.mark_finished(
                output_text
            )

            return request

        except Exception as exc:

            if request.status.value in (
                "waiting",
                "running"
            ):

                request.mark_failed(
                    str(exc)
                )

            raise


    def generate_batch(
        self,
        batch
    ):

        results = []


        for request in batch:

            try:

                result = self.generate(
                    request
                )

                results.append(
                    result
                )

            except Exception:

                # 单个请求失败不能让整个 batch
                # 都停止执行
                results.append(
                    request
                )


        return results


    def step(
        self,
        requests
    ):

        activate_results = []

        for request in requests:

            # 第一次进入 decode
            if request.status == RequestStatus.WAITING:

                request.mark_prefill()

            request.mark_decoding()

            # 模拟生成一个 token

            token = (
                f"token_{request.current_step}"
            )

            request.append_token(
                token
            )

            # 达到最大长度

            if request.is_finished:

                output = "".join(
                    request.generated_tokens
                )

                request.mark_finished(
                    output
                )

            else:

                activate_results.append(
                    request
                )


        return activate_results
