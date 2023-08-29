import io
import math
import socket
import threading
import time
import queue
import audioop
import collections
import whisper
import resampy
import soundfile as sf
import speech_recognition as sr

from torch import cuda
from datetime import datetime, timedelta
from numpy import float32 as FLOAT32

from src.utils.exceptions import WaitTimeoutError
from src.utils.model_parameters import ModelSize


SAMPLING_RATE_16K: int = 16000


class TranscriptSocket:
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
        model_size: ModelSize = ModelSize.BASE,
    ) -> None:
        # @todo: check if necessary
        self._cb: callable = callback

        assert bit_depth in [8, 16, 24, 32], "Provided bit-depth is invalid."
        assert sample_rate in [
            16e3,
            44.1e3,
            48e3,
            96e3,
        ], "Provided sample rate unsupported."
        assert isinstance(is_stereo, bool), "Parameter 'is_stereo' is of invalid type."
        self._data_chunk: int = data_chunk
        self._bit_depth: int = bit_depth
        self._sample_width: int = bit_depth // 8  # Could also be called byte_depth
        self._sample_rate: int = sample_rate
        self._stereo: bool = is_stereo
        self._energy_threshold = 300  # minimum audio energy to consider for recording
        self._audiodata_queue: queue.Queue = queue.Queue()

        # Prepare transcription model
        # --- Define model type (size)
        self._whisper_model_type: ModelSize = model_size
        model_type_s: str = self._whisper_model_type.value
        if self._whisper_model_type != ModelSize.LARGE:
            model_type_s = model_type_s + ".en"
        # --- Load model
        print("Loading model...", end=" ", flush=True)
        self._whisper_model = whisper.load_model(model_type_s)
        print("Model loaded.")

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

        # Receive audio data in background make pass it through sentence detector
        listener_stopper_func: callable = self._listen_in_background(
            self._client_socket,
            phrase_time_limit=1.0,
        )
        self._transcribe(exit_event)
        # Stop listening in background
        listener_stopper_func()

    def _listen(
        self,
        source: socket.socket,
        timeout: int | None = None,
        phrase_time_limit: int | None = None,
    ) -> sr.AudioData:
        assert (
            TranscriptSocket.PAUSE_THRESHOLD
            >= TranscriptSocket.NON_SPEAKING_DURATION
            >= 0
        )

        i = 0  # @debug
        source.settimeout(None)

        seconds_per_buffer: float = float(self._data_chunk) / self._sample_rate
        pause_buffer_count: int = int(
            math.ceil(TranscriptSocket.PAUSE_THRESHOLD / seconds_per_buffer)
        )  # number of buffers of non-speaking audio during a phrase, before the phrase should be considered complete
        phrase_buffer_count: int = int(
            math.ceil(TranscriptSocket.PHRASE_THRESHOLD / seconds_per_buffer)
        )  # minimum number of buffers of speaking audio before we consider the speaking audio a phrase
        non_speaking_buffer_count: int = int(
            math.ceil(TranscriptSocket.NON_SPEAKING_DURATION / seconds_per_buffer)
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

    def _listen_in_background(
        self,
        source: socket.socket,
        phrase_time_limit: int | None = None,
    ):
        assert isinstance(source, socket.socket), "Source must be a connected socket"
        running = [True]

        def threaded_listen():
            while running[0]:
                try:  # listen for 1 second, then check again if the stop function has been called
                    audio = self._listen(source, 1, phrase_time_limit)
                except (
                    WaitTimeoutError,
                    TimeoutError,
                ):  # listening timed out, just try again
                    continue
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

    # @todo: define "phrase_timeout"
    def _transcribe(self, exit_event: threading.Event, phrase_timeout: float = 3.0):
        # The last time a recording was retreived from the queue: @see self._audiodata_queue below
        phrase_time = None
        # Current raw audio bytes.
        last_sample: bytes = bytes()
        # @note: Yes the empty string is important
        self._transcripts: list[str] = [""]

        # Phrase identifier (incrementing each new sentence)
        phrase_id: int = 0
        # Phrase version (incrementing each new version of the same sentence)
        # --- Resets to 0 when new sentence detected
        phrase_version: int = 0

        while not exit_event.is_set():
            now: datetime = datetime.utcnow()

            # Pull raw recorded audio from the queue.
            if not self._audiodata_queue.empty():
                phrase_complete: bool = False

                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(
                    seconds=phrase_timeout
                ):
                    last_sample = bytes()
                    phrase_complete = True

                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not self._audiodata_queue.empty():
                    data = self._audiodata_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(
                    frame_data=last_sample,
                    sample_rate=self._sample_rate,
                    sample_width=self._sample_width,
                )
                wav_stream = io.BytesIO(audio_data.get_wav_data())
                audio_data, origin_sampling_rate = sf.read(wav_stream)
                audio_data = resampy.resample(
                    audio_data,
                    origin_sampling_rate,
                    SAMPLING_RATE_16K,  # This is the required sampling rate for transcribing with whisper.
                )
                audio_data = audio_data.astype(FLOAT32)

                # Read the transcription.
                result = self._whisper_model.transcribe(
                    audio_data, fp16=cuda.is_available()
                )
                text: str = result["text"].strip()

                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    phrase_id += 1
                    phrase_version = 0
                    self._transcripts.append(text)
                else:
                    phrase_version += 1
                    self._transcripts[-1] = text
                self._cb(self._transcripts[-1], phrase_id, phrase_version)

                # Infinite loops are bad for processors, must sleep.
                time.sleep(250e-3)  # Here: 250ms

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
