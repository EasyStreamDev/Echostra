# import pyaudio

# p = pyaudio.PyAudio()
# device_count = p.get_device_count()

# for i in range(device_count):
#     device_info = p.get_device_info_by_index(i)
#     device_name = device_info["name"]
#     device_description = device_info["hostApi"] if "hostApi" in device_info else ""
#     device_max_input_channels = device_info["maxInputChannels"]

#     if device_max_input_channels > 0:
#         print(f"Device {i}: {device_name} ({device_description})")

# p.terminate()

import speech_recognition as sr

print(sr.Microphone(device_index=0))
