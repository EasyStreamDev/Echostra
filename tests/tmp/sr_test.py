import speech_recognition as sr
import pyaudio
from pprint import pprint

init_rec = sr.Recognizer()
with sr.Microphone() as source:
    print("Let's speak!!")
    audio_data = init_rec.record(source, duration=10)
    print("Recognizing your text.............")
    text = init_rec.recognize_whisper(audio_data)
    print(text)
# pprint(sr.Microphone.list_microphone_names())
# pprint(sr.Microphone.list_working_microphones())
