import pyaudio

CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # Format of audio data
CHANNELS = 1  # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100  # Sample rate (samples per second)


p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    output=True,  # Set to False if you're creating an input stream
)
