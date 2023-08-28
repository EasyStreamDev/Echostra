import os
import io

import whisper
import soundfile as sf
import speech_recognition as sr

from time import sleep
from queue import Queue
from datetime import datetime, timedelta
from typing import Callable, Union, List
from torch import cuda
from numpy import float32 as FLOAT32

from src.transcription.model_parameters import ModelSize


DEFAULT_OUTPUT_FILE = "transcript.log"


# Default callback on transcription new results
def default_callback(_transcript: list[str]):
    for _s in _transcript:
        print(_s)


class RealTimeTranscriptor:
    def __init__(
        self,
        model_size: ModelSize = ModelSize.BASE,
        energy_th: int = 1000,
        record_timeout: float = 2,
        phrase_timeout: float = 3,
    ) -> None:
        self.model_type: ModelSize = model_size
        self.energy_threshold: int = energy_th
        self.record_timeout: float = record_timeout
        self.phrase_timeout: float = phrase_timeout
        self.running: bool = False
        self.transcripts: list[str] = [""]

        model_type_s: str = self.model_type.value
        # Load transcription model
        if self.model_type != ModelSize.LARGE:
            model_type_s = model_type_s + ".en"
        print("Loading model...", end=" ")
        self.model = whisper.load_model(model_type_s)
        print("Model loaded.")

    def run(
        self,
        callback: Callable[[List[str]], None] = None,
        output_file: Union[str, None] = None,
    ) -> None:
        """Runs the transcription model
        Also creates a thread listening to the specified audio source

        Args:
            mic_index (int): Index of the microphone to listen to
            callback (Callable[[List[str]]], optional): Function called when new transcription is inferred. Defaults to None.
            output_file (Union[str, None], optional): File where the final transcription will be written when the transcription is shut down. Defaults to None.
        """

        self.running = True

        # Parameters check
        if callback == None:
            callback = default_callback
        if output_file == None:
            output_file = DEFAULT_OUTPUT_FILE

        # The last time a recording was retreived from the queue: @see data_queue below
        phrase_time = None
        # Current raw audio bytes.
        last_sample = bytes()
        # Thread safe Queue for passing data from the threaded recording callback.
        data_queue = Queue()

        SAMPLE_RATE: int
        SAMPLE_WIDTH: int

        # @todo: set dynamic_energy_threshold to False
        # @todo: check sr.Recognizer().adjust_for_ambient_noise

        def __record_callback(_, audio_data: sr.AudioData):
            """
            Threaded callback function to recieve audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            # Grab the raw bytes and push it into the thread safe queue.
            data_queue.put(audio_data.get_raw_data())

        SAMPLE_RATE: int  # @dev: Needs to be set
        SAMPLE_WIDTH: int  # @dev: Needs to be set

        while self.running:
            try:
                now: datetime = datetime.utcnow()
                # Pull raw recorded audio from the queue.
                if not data_queue.empty():
                    phrase_complete: bool = False
                    # If enough time has passed between recordings, consider the phrase complete.
                    # Clear the current working audio buffer to start over with the new data.
                    if phrase_time and now - phrase_time > timedelta(
                        seconds=self.phrase_timeout
                    ):
                        last_sample = bytes()
                        phrase_complete = True
                    # This is the last time we received new audio data from the queue.
                    phrase_time = now

                    # Concatenate our current audio data with the latest audio data.
                    while not data_queue.empty():
                        data = data_queue.get()
                        last_sample += data

                    # Use AudioData to convert the raw data to wav data.
                    audio_data = sr.AudioData(
                        frame_data=last_sample,
                        sample_rate=SAMPLE_RATE,
                        sample_width=SAMPLE_WIDTH,
                    )
                    wav_stream = io.BytesIO(audio_data.get_wav_data())
                    audio_array, sampling_rate = sf.read(wav_stream)
                    audio_array = audio_array.astype(FLOAT32)

                    # Read the transcription.
                    result = self.model.transcribe(
                        audio_array, fp16=cuda.is_available()
                    )
                    text = result["text"].strip()

                    # If we detected a pause between recordings, add a new item to our transcripion.
                    # Otherwise edit the existing one.
                    if phrase_complete:
                        self.transcripts.append(text)
                    else:
                        self.transcripts[-1] = text
                    callback(self.transcripts)

                    # Infinite loops are bad for processors, must sleep.
                    sleep(0.25)
            except KeyboardInterrupt:
                self.running = False

        # On ending, save transcripts to file
        with open(output_file, "w+") as f:
            for _s in self.transcripts:
                f.write(_s + "\n")


if __name__ == "__main__":
    from pprint import pprint

    def _cb1(transcripts):
        # Clear the console to reprint the updated transcription.
        os.system("cls" if os.name == "nt" else "clear")
        for line in transcripts:
            print(line)
        # Flush stdout
        print("", end="", flush=True)

    t = RealTimeTranscriptor(model_size=ModelSize.BASE)

    t.run(mic_index=None, callback=_cb1, output_file="output.log")
