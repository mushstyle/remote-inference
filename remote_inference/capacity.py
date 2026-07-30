"""Bounded execution capacity for synchronous inference work."""
from contextlib import contextmanager
from threading import BoundedSemaphore, Lock
from typing import Iterator


class BoundedWorkQueue:
    """Limit active work and reject requests beyond a bounded backlog."""

    def __init__(self, concurrency: int, max_in_flight: int):
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if max_in_flight < concurrency:
            raise ValueError("max_in_flight must be greater than or equal to concurrency")

        self._slots = BoundedSemaphore(concurrency)
        self._count_lock = Lock()
        self._in_flight = 0
        self._max_in_flight = max_in_flight

    @contextmanager
    def reserve(self) -> Iterator[bool]:
        """Reserve queue capacity and an execution slot when available."""
        with self._count_lock:
            if self._in_flight >= self._max_in_flight:
                accepted = False
            else:
                self._in_flight += 1
                accepted = True

        if not accepted:
            yield False
            return

        try:
            with self._slots:
                yield True
        finally:
            with self._count_lock:
                self._in_flight -= 1
