import math
import socket
import threading
import collections
import queue
import audioop

import speech_recognition as sr

from src.utils.exceptions import WaitTimeoutError


class StreamSocket:
    PAUSE_THRESHOLD: float = 0.8
    PHRASE_THRESHOLD: float = 0.3
    NON_SPEAKING_DURATION: float = 0.5

    def __init__(
        self,
        callback: callable,
        bit_depth: int,
        sample_rate: int,
        is_stereo: bool = False,
        data_chunk: int = 1024,
    ) -> None:
        # @todo: check if necessary
        self._cb: callable = callback

        assert bit_depth in [8, 16, 24, 32], "Provided bit-depth is invalid."
        assert sample_rate in [44.1e3, 48e3, 96e3], "Provided sample rate unsupported."
        assert isinstance(is_stereo, bool), "Parameter 'is_stereo' is of invalid type."
        self._data_chunk: int = data_chunk
        self._bit_depth: int = bit_depth
        self._sample_width: int = bit_depth // 8
        self._sample_rate: int = sample_rate
        self._stereo: bool = is_stereo
        self._energy_threshold = 300  # minimum audio energy to consider for recording

        self._audiodata_queue: queue.Queue = queue.Queue()

        self._socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client_socket: socket.socket | None = None

        self._socket.bind(("localhost", 0))
        self._socket.settimeout(250e-3)  # Timeout set to 250ms
        self._port: int = self._socket.getsockname()[1]
        self._socket.listen()
        self._running: bool = True
        self._exit_event: threading.Event | None = None

    def start(self, exit_event: threading.Event):
        # Event closing loops when set
        self._exit_event = exit_event

        # Loop until connection to the opened socket is found.
        while not exit_event.is_set():
            try:
                self._client_socket, _ = self._socket.accept()
                break
            except TimeoutError:
                continue

        listener_stopper_func: callable = self.listen_in_background(self._client_socket)
        while not exit_event.is_set():
            try:
                pass
            except WaitTimeoutError:
                continue
            except TimeoutError:
                continue

        # Stop listening in background
        listener_stopper_func()

    def listen(
        self,
        source: socket.socket,
        timeout: int | None = None,
        phrase_time_limit: int | None = None,
    ) -> sr.AudioData:
        assert StreamSocket.PAUSE_THRESHOLD >= StreamSocket.NON_SPEAKING_DURATION >= 0

        i = 0  # @debug
        source.settimeout(None)

        seconds_per_buffer: float = float(self._data_chunk) / self._sample_rate
        pause_buffer_count: int = int(
            math.ceil(StreamSocket.PAUSE_THRESHOLD / seconds_per_buffer)
        )  # number of buffers of non-speaking audio during a phrase, before the phrase should be considered complete
        phrase_buffer_count: int = int(
            math.ceil(StreamSocket.PHRASE_THRESHOLD / seconds_per_buffer)
        )  # minimum number of buffers of speaking audio before we consider the speaking audio a phrase
        non_speaking_buffer_count: int = int(
            math.ceil(StreamSocket.NON_SPEAKING_DURATION / seconds_per_buffer)
        )  # maximum number of buffers of non-speaking audio to retain before and after a phrase

        elapsed_time = 0  # number of seconds of audio read
        buffer: bytes = b""  # an empty buffer means that the stream has ended and there is no data left to read

        while True:
            frames = collections.deque()

            while True:
                # handle waiting too long for phrase by raising an exception
                elapsed_time += seconds_per_buffer
                if timeout and elapsed_time > timeout:
                    raise WaitTimeoutError(
                        "listening timed out while waiting for phrase to start"
                    )

                buffer = source.recv(self._data_chunk)
                if len(buffer) == 0:
                    break  # reached end of the stream
                frames.append(buffer)
                if (
                    len(frames) > non_speaking_buffer_count
                ):  # ensure we only keep the needed amount of non-speaking buffers
                    frames.popleft()

                # print(f"[{len(buffer)}][{i}] Silence detector")
                energy: int = audioop.rms(
                    buffer, self._sample_width
                )  # @todo: compute sample width from audio parameters
                if energy > self._energy_threshold:
                    print("Sentence detected... ", end="", flush=True)
                    break

            # read audio input until the phrase ends
            pause_count, phrase_count = 0, 0
            phrase_start_time = elapsed_time
            while True:
                # handle phrase being too long by cutting off the audio
                elapsed_time += seconds_per_buffer
                if (
                    phrase_time_limit
                    and elapsed_time - phrase_start_time > phrase_time_limit
                ):
                    break

                buffer = source.recv(self._data_chunk)
                if len(buffer) == 0:
                    break  # reached end of the stream
                frames.append(buffer)
                phrase_count += 1

                # check if speaking has stopped for longer than the pause threshold on the audio input
                # print(f"[{len(buffer)}][{i}] Speech detector energy")
                energy = audioop.rms(
                    buffer, self._sample_width
                )  # unit energy of the audio signal within the buffer
                if energy > self._energy_threshold:
                    pause_count = 0
                else:
                    pause_count += 1
                if pause_count > pause_buffer_count:  # end of the phrase
                    print("Finished.")
                    break

            # check how long the detected phrase is, and retry listening if the phrase is too short
            phrase_count -= (
                pause_count  # exclude the buffers for the pause before the phrase
            )
            if phrase_count >= phrase_buffer_count or len(buffer) == 0:
                break  # phrase is long enough or we've reached the end of the stream, so stop listening

        # obtain frame data
        for i in range(pause_count - non_speaking_buffer_count):
            frames.pop()  # remove extra non-speaking frames at the end
        frame_data = b"".join(frames)

        return sr.AudioData(frame_data, self._sample_rate, self._sample_width)

    def listen_in_background(
        self,
        source: socket.socket,
        phrase_time_limit: int | None = None,
    ):
        assert isinstance(source, socket.socket), "Source must be a connected socket"
        running = [True]

        def threaded_listen():
            while running[0]:
                try:  # listen for 1 second, then check again if the stop function has been called
                    audio = self.listen(source, 1, phrase_time_limit)
                except WaitTimeoutError:  # listening timed out, just try again
                    pass
                else:
                    if running[0]:
                        self._audio_data_received_callback(audio)

        def stopper(wait_for_stop=True):
            running[0] = False
            if wait_for_stop:
                listener_thread.join()  # block until the background thread is done, which can take around 1 second

        listener_thread = threading.Thread(target=threaded_listen)
        listener_thread.daemon = True
        listener_thread.start()
        return stopper

    def is_running(self) -> bool:
        return False if not self._exit_event or not self._exit_event.is_set() else True

    def get_port(self) -> int:
        return self._port

    def _audio_data_received_callback(self, data: sr.AudioData):
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        self._audiodata_queue.put(data.get_raw_data())
        print("--------------------------")
