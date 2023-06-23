import sys
import pyaudio


class _paHostApiTypeId:
    paInDevelopment = 0  # use while developing support for a new host API */
    paDirectSound = 1
    paMME = 2
    paASIO = 3
    paSoundManager = 4
    paCoreAudio = 5
    paOSS = 7
    paALSA = 8
    paAL = 9
    paBeOS = 10
    paWDMKS = 11
    paJACK = 12
    paWASAPI = 13
    paAudioScienceHPI = 14
    paAudioIO = 15


def get_os_type() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    elif sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform.startswith("darwin"):
        return "macos"
    # else
    return "unknown OS"


def _list_microphones(host_api: dict, p: pyaudio.PyAudio) -> list[dict]:
    device_count: int = host_api["deviceCount"]
    available_mics: list[dict] = []

    for i in range(device_count):
        device_info = p.get_device_info_by_host_api_device_index(host_api["index"], i)
        if device_info["maxInputChannels"] > 0:
            available_mics.append(device_info)

    return available_mics


def _get_win_audio_host_api(p: pyaudio.PyAudio):
    return p.get_host_api_info_by_type(_paHostApiTypeId.paWASAPI)


def _get_linux_audio_host_api(p: pyaudio.PyAudio):
    return p.get_host_api_info_by_type(_paHostApiTypeId.paALSA)


def _get_macos_audio_host_api(p: pyaudio.PyAudio):
    return p.get_host_api_info_by_type(_paHostApiTypeId.paCoreAudio)


def get_mics_list():
    p = pyaudio.PyAudio()
    OS: str = get_os_type()

    # Define audio host API depending on OS
    if OS == "windows":
        host_api_data = _get_win_audio_host_api(p=p)
    elif OS == "linux":
        host_api_data = _get_linux_audio_host_api(p=p)
    else:
        p.terminate()
        raise Exception("Unsupported OS")

    mics_list = _list_microphones(host_api_data, p=p)
    p.terminate()

    return mics_list


if __name__ == "__main__":
    import pprint

    pprint.pprint(get_mics_list())
