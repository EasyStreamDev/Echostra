import queue

from src.utils.ESThread import ESThread
from speech_recognition import AudioSource
from network.AudioStreamSocket import StreamSocket


class ESAudioStream(AudioSource):
    def __init__(self):
        self._listening: bool = False
        self._data_queue: queue.Queue = queue.Queue()

    def __enter__(self):
        self._listening = True

    def __exit__(self):
        self._listening = False

    def write(self, data: bytes):
        # @todo: format data ?
        if self._listening:
            self._data_queue.put(data)

    def read(self) -> bytes:
        return self._data_queue.get()
