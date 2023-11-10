import io
import enum
import math
import time
import json
import socket
import audioop
import whisper
import threading
import soundfile as sf
from typing import Any
from torch import cuda
from time import sleep
from queue import Queue
from pprint import pprint
from collections import deque
import speech_recognition as sr
from numpy import float32 as FLOAT32
from datetime import datetime, timedelta
from resampy import resample as py_resample
from collections.abc import Callable, Iterable, Mapping


SAMPLING_RATE_16K: int = 16000

class ModelSize(enum.Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class WaitTimeoutError(Exception):
    pass


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
        self._audiodata_queue: Queue = Queue()

        # Prepare transcription model
        # --- Define model type (size)
        self._whisper_model_type: ModelSize = model_size
        model_type_s: str = self._whisper_model_type.value
        if self._whisper_model_type != ModelSize.LARGE:
            model_type_s = model_type_s
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

        print(cuda.is_available())
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

        exit_condition: bool = False
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

        while not exit_condition:
            frames = deque()

            while True:
                # handle waiting too long for phrase by raising an exception
                elapsed_time += seconds_per_buffer
                if timeout and elapsed_time > timeout:
                    raise WaitTimeoutError(
                        "listening timed out while waiting for phrase to start"
                    )

                buffer = source.recv(self._data_chunk)
                if len(buffer) == 0:
                    if not self._exit_event.is_set():
                        self._exit_event.set()
                    break  # reached end of the stream
                # print("Data received: silence")
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
                    if not self._exit_event.is_set():
                        self._exit_event.set()
                    break  # reached end of the stream
                # print("Data received: sentence")
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
        for _ in range(pause_count - non_speaking_buffer_count):
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
        print("\t--- Listening to audio data.")
        return stopper

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

        # index = 0

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
                print(f"Size of audio to transcribe: {len(last_sample)}")

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(
                    frame_data=last_sample,
                    sample_rate=self._sample_rate,
                    sample_width=self._sample_width,
                )

                wav_stream = io.BytesIO(audio_data.get_wav_data())
                audio_data, origin_sampling_rate = sf.read(wav_stream)

                # if phrase_complete:
                #     index += 1
                # sf.write(
                #     f"output_{index}.wav",
                #     audio_data,
                #     origin_sampling_rate,
                # )

                audio_data = py_resample(
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


class _ThreadSafeDict:
    def __init__(self, val: dict | None = None):
        self._dict = val if val != None else dict()
        self._lock = threading.Lock()

    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]

    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)

    def keys(self):
        with self._lock:
            return list(self._dict.keys())

    def values(self):
        with self._lock:
            return list(self._dict.values())

    def items(self):
        with self._lock:
            return list(self._dict.items())


class TCPServer:
    CONNECTION_SUCCESS_MSG = json.dumps(
        {"status_code": "200", "message": "Connected"}
    ).encode()

    def __init__(self, host: str, port: int) -> None:
        # Create a socket object
        self._server_socket: socket.socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )
        # Create the list that will contain all clients
        self._clients: _ThreadSafeDict = _ThreadSafeDict()
        # Set the variable in control of the server running.
        self._running: bool = False

        # Storing host and port
        self._host, self._port = host, port
        # Bind the socket to a specific host and port
        self._server_socket.bind((self._host, self._port))
        # Set the timeout for accepting client connections
        self._server_socket.settimeout(50e-3)  # Timeout set to 50ms

    # Server main function (blocking)
    def run(self) -> None:
        # Listen for incoming connections
        self._server_socket.listen()
        self._running = True

        print(f"TCP Server listening on {self._host}:{self._port}")

        # Accept incoming connections in the main thread
        while self._running:
            try:
                # Accept client connection
                client_socket, client_address = self._server_socket.accept()
                # Handle the client connection in a separate thread
                client_thread = ESThread(
                    target=self._handle_client_connect,
                    args=(client_socket, client_address),
                )
                # Add client and related thread to client list
                self._clients[client_address] = {
                    "socket": client_socket,
                    "esthread": client_thread,
                }
                # Start client thread
                client_thread.start()
            except TimeoutError:
                self._remove_closed_connections()
                continue
            except socket.error as e:
                # Handle other socket errors
                print("Error occurred in server main thread:", e)
                break

        # Tell every connection to stop functionning
        for key in self._clients:
            w: ESThread = self._clients.get(key)["esthread"]
            w.stop()
        # Waiting for all clients thread to terminate.
        while self._clients.items().__len__() != 0:
            self._remove_closed_connections()
            sleep(250e-3)  # Sleep for 250ms

    # Function to handle client connections
    def _handle_client_connect(self, client_socket: socket.socket, client_address: str):
        global COMMAND_TO_METHOD

        print(f"Connected to client: {client_address}")
        this_thread: ESThread = self._clients.get(client_address)["esthread"]

        print("Sending connection success message.")
        client_socket.sendall(TCPServer.CONNECTION_SUCCESS_MSG)
        print("Sent!")
        # Receive and process client data
        while not this_thread.get_exit_event().is_set():
            print("Waiting data from client.")
            msg = client_socket.recv(1024)
            if not msg:
                break
            print(f"Received: {msg.decode()}")

            try:
                # Process the received message
                data: dict = json.loads(msg.decode())

                COMMAND_TO_METHOD.get(
                    data.get("command", None), TCPServer._cmd_default_handler
                )(self, this_thread, data, client_socket)
            except:  # Ignore any error not supported
                pass

        # Close the connection
        client_socket.close()
        print(f"Connection closed with client: {client_address}")

    def _remove_closed_connections(self):
        """
        Removes closed connections from the `_clients` dictionary.

        It iterates over the `_clients` dictionary and checks if each client's socket is still connected and if its associated thread is still alive.
        - If both conditions are met, the client is considered to be staying, and it is added to the `staying_clients` dictionary.
        - If either the socket is closed or the thread is not alive, the client is considered to be disconnected. Its associated exit event is set to signal the thread to exit, and it is added to the `disconnected_clients` dictionary.

        After iterating over all clients, the `_clients` dictionary is updated with the `staying_clients` dictionary, which contains only the staying clients.

        Finally, the disconnected clients' threads are joined to wait for their completion.

        Note: This function assumes that `_clients` is a dictionary with client information, where the key is the client identifier and the value is a dictionary containing the client's socket, thread, and exit event.
        """
        staying_clients = _ThreadSafeDict()
        disconnected_clients = dict()

        for key, value in self._clients.items():
            if value["socket"].fileno() != -1 and value["esthread"].is_alive():
                staying_clients[key] = value
            else:
                w: ESThread = value["esthread"]
                w.stop()
                disconnected_clients[key] = value

        self._clients = staying_clients

        for key, value in disconnected_clients.items():
            value["esthread"].join()

    def _cmd_default_handler(self, *args):
        pass

    def _cmd_createSTTStream(
        self, this_thread: ESThread, data: dict, client_socket: socket.socket
    ):
        try:
            req_params: dict = data.get("params")

            def _send_result_callback(
                transcript: str,
                id: int,
                version: int,
            ):
                try:
                    print("\t--- Sending transcription results.")
                    client_socket.sendall(
                        json.dumps(
                            {
                                "type": "transcript",
                                "mic_id": req_params.get("mic_id", None),
                                "transcript": transcript,
                                "phrase_id": id,
                                "phrase_version": version,
                            }
                        ).encode()
                    )
                    print("\t--- Sent")
                except:
                    pass

            print("\t--- Creating transcription stream socket")
            stream_socket: TranscriptSocket = TranscriptSocket(
                callback=_send_result_callback,
                bit_depth=req_params.get("bit_depth"),
                sample_rate=req_params.get("sample_rate"),
                is_stereo=req_params.get("stereo"),
            )
            print("\t--- Creating new thread")
            stream_thread_exit_event: threading.Event = threading.Event()
            stream_thread: ESThread = ESThread(
                target=stream_socket.start,
                exit_event=stream_thread_exit_event,
                args=(stream_thread_exit_event,),
            )
            print("\t--- Starting new thread")
            stream_thread.start()
            this_thread.add_subthread(stream_thread)

            print("\t--- Notifying requester of success")
            # 3. Inform client socket about which port it should connect on.
            client_socket.sendall(
                json.dumps(
                    {
                        "type": "createSTTStream",
                        "mic_id": req_params.get("mic_id"),
                        "message": "OK, ready to receive data.",
                        "statusCode": 201,
                        "port": stream_socket.get_port(),
                    }
                ).encode(),
            )
            print("\t--- Transcription launched!")
        except socket.error:
            client_socket.sendall(
                json.dumps(
                    {
                        "type": "createSTTStream",
                        "mic_id": req_params.get("mic_id"),
                        "message": "Failure, unable to perform request.",
                        "statusCode": 500,
                    }
                ).encode(),
            )


COMMAND_TO_METHOD = {
    "createSTTStream": TCPServer._cmd_createSTTStream,
}

HOST = "127.0.0.1"  # Localhost
PORT = 47921  # Port number

server = TCPServer(host=HOST, port=PORT)
server.run()