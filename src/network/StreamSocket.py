import socket
import threading


class StreamSocket:
    def __init__(self, callback: callable) -> None:
        self._cb: callable = callback
        self._running: bool = True
        self._socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client_socket: socket.socket | None = None

        self._socket.bind(("localhost", 0))
        self._socket.settimeout(250e-3)  # Timeout set to 250ms
        self._port: int = self._socket.getsockname()[1]
        self._socket.listen()

    def start(self, exit_event: threading.Event):
        while not exit_event.is_set():
            try:
                self._client_socket, _ = self._socket.accept()
                break
            except TimeoutError:
                continue

        while not exit_event.is_set():
            try:
                # @note: if blocking, may not leave when told to
                data: bytes = self._client_socket.recv(1024)
                if not data:
                    break
                self._cb(data)
            except TimeoutError:
                continue

    def get_port(self) -> int:
        return self._port
