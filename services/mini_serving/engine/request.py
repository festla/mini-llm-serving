import time
import uuid

from dataclasses import dataclass, field
from enum import Enum
from threading import Event

class RequestStatus(str, Enum):

    WAITING = "waiting"

    PREFILL = "prefill"

    RUNNING = "running"

    DECODING = "decoding"

    FINISHED = "finished"

    FAILED = "failed"


@dataclass
class Request:

    prompt: str

    max_tokens: int = 50

    request_id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )

    status: RequestStatus = RequestStatus.WAITING

    created_at: float = field(
        default_factory=time.time
    )

    finished_event: Event = field(
        default_factory=Event,
        repr=False
    )

    started_at: float | None = None

    finished_at: float | None = None

    output_text: str | None = None

    generated_tokens: list[str] = field(
        default_factory=list
    )

    current_step: int = 0

    error: str | None = None

    def __post_init__(self):

        if not self.prompt.strip():

            raise ValueError(
                "prompt must not be a empty"
            )

        if self.max_tokens <= 0:

            raise ValueError(
                "max_tokens must be greater than 0"
            )

    def mark_running(self):

        if self.status != RequestStatus.WAITING:

            raise ValueError(
                f"cannot transition from "
                f"{self.status} to running"
            )

        self.status = RequestStatus.RUNNING

        self.started_at = time.time()


    def mark_prefill(self):

        if self.status != RequestStatus.WAITING:

            raise ValueError(
                "request is not waiting"
            )


        self.status = RequestStatus.PREFILL

        self.started_at = time.time()  


    def mark_decoding(self):

        if self.status not in (
            RequestStatus.PREFILL,
            RequestStatus.RUNNING,
            RequestStatus.DECODING
        ):

            raise ValueError(
                "invalid decoding transition"
            )


        self.status = RequestStatus.DECODING


    def mark_finished(
        self,
        output_text: str
    ):

        if self.status not in (
            RequestStatus.RUNNING,
            RequestStatus.DECODING
        ):

            raise ValueError(
                f"cannot transition from "
                f"{self.status} to finished"
            )

        self.status = RequestStatus.FINISHED

        self.output_text = output_text

        self.finished_at = time.time()
        self.finished_event.set()

    def mark_failed(
        self,
        error: str
    ):
        if self.status not in (
            RequestStatus.WAITING,
            RequestStatus.RUNNING
        ):

            raise ValueError(
                f"cannot transition from "
                f"{self.status} to failed"
            )

        self.status = RequestStatus.FAILED

        self.error = error

        self.finished_at = time.time()
        
        self.finished_event.set()
        

    def append_token(
        self,
        token: str
    ):

        self.generated_tokens.append(
            token
        )

        self.current_step += 1


    @property
    def queue_time(self):

        if self.started_at is None:
            return None

        return (
            self.started_at
            -
            self.created_at
        )


    @property
    def inference_time(self):

        if (
            self.started_at is None
            or self.finished_at is None
        ):
            return None

        return (
            self.finished_at
            -
            self.started_at
        )


    @property
    def total_latency(self):

        if self.finished_at is None:
            return None

        return (
            self.finished_at
            -
            self.created_at
        )


    @property
    def is_finished(self):

        return (
            self.current_step
            >=
            self.max_tokens
        )


