from collections.abc import Callable, Iterable, Mapping
import threading
from typing import Any


class ESThread(threading.Thread):
    def __init__(
        self,
        group: None = None,
        target: Callable[..., object] | None = None,
        name: str | None = None,
        exit_event: threading.Event | None = None,
        args: Iterable[Any] = ...,
        kwargs: Mapping[str, Any] | None = None,
        *,
        daemon: bool | None = None,
    ) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self._exit_event: threading.Event = (
            exit_event if exit_event != None else threading.Event()
        )
        self._subthreads: list[ESThread] = []

    def stop(self):
        # Notifying all subthreads to end their execution politely
        for st in self._subthreads:
            st.stop()

        # Cut this thread main loop
        self._exit_event.set()

        for st in self._subthreads:
            st.join()

    def add_subthread(self, thread: "ESThread") -> bool:
        type_valid: bool = bool(type(thread) == ESThread)

        if type_valid:
            self._subthreads.append(thread)
        return type_valid

    def get_exit_event(self) -> threading.Event:
        return self._exit_event
