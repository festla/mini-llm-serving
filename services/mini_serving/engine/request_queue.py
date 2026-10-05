from collections import deque
from threading import Lock

from services.mini_serving.engine.request import (
    Request
)


class RequestQueue:

    def __init__(self):

        self._queue = deque()

        self._lock = Lock()



    def enqueue(
        self,
        request: Request
    ):

        with self._lock:

            self._queue.append(
                request
            )


    def dequeue(self):

        with self._lock:

            if not self._queue:

                return None

            return self._queue.popleft()


    def size(self):

        with self._lock:

            return len(
                self._queue
            )


    def empty(self):

        return self.size() == 0
