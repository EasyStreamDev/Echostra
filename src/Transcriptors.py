from typing import Callable, Union, List
from model_parameters import ModelSize


DEFAULT_OUTPUT_FILE = "transcript.log"


# Default callback on transcription new results
def default_callback(_transcript: list[str]):
    for _s in _transcript:
        print(_s)


class WhisperTranscriptor:
    def __init__(
        self,
        model_size: ModelSize = "base",
        energy_th: int = 1000,
        record_timeout: float = 2,
        phrase_timeout: float = 3,
    ) -> None:
        self.model_size: ModelSize = model_size
        self.energy_threshold: int = energy_th
        self.record_timeout: float = record_timeout
        self.phrase_timeout: float = phrase_timeout
        self.transcripts: list[str] = []

    # Blocking
    def run(
        self,
        callback: Callable[[List[str]]] = None,
        output_file: Union[str, None] = None,
    ):
        # Parameters check
        if callback == None:
            callback = default_callback
        if output_file == None:
            output_file = DEFAULT_OUTPUT_FILE

        # On ending, save transcripts to file
        with open(output_file, "w+") as f:
            for _s in self.transcripts:
                f.write(_s)

    def reset(self):
        self.transcripts: list[str] = []
