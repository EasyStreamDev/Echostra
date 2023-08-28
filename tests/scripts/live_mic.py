import socket
import time
import json

import speech_recognition as sr

HOST = "127.0.0.1"
PORT = 47921  # The same port as used by the server


def send(s: socket.socket, data: bytes):
    s.sendall(data)


def receive(s: socket.socket, display: bool = False) -> str:
    response = s.recv(2048).decode()
    if display:
        print("Server response: ", end="")
        print(response)
    return response


def connect(s: socket.socket, host: str, port: int, await_confirmation: bool = True):
    s.connect((host, port))
    if await_confirmation:
        receive(s, display=True)


if __name__ == "__main__":
    # Create TCP socket
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to server
    connect(s=tcp_socket, host=HOST, port=PORT)
    # Send request for creation of a transcription stream
    send(
        tcp_socket,
        json.dumps(
            {
                "command": "createSTTStream",
                "params": {
                    "mic_id": "",
                    "bit_depth": 16,
                    "sample_rate": 48e3,
                    "stereo": False,
                },
            }
        ).encode(),
    )
    # Get response
    response: dict = json.loads(receive(tcp_socket))

    try:
        # Check if connection request was accepted
        if response.get("statusCode", 404) == 201:
            # Get port to connect to in response
            port: int = response.get("port")
            # Prepare audio data stream socket
            stream_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Connect to stream socket endpoint
            connect(
                s=stream_socket,
                host=HOST,
                port=port,
                await_confirmation=False,
            )

            try:
                with sr.Microphone(sample_rate=48000) as source:
                    i = 0
                    while True:
                        buffer: bytes = source.stream.read(1024)
                        if len(buffer) == 0:
                            break
                        send(stream_socket, buffer)
            except KeyboardInterrupt:
                pass
    except Exception as e:
        print(f"Did not work properly somewhere:\n{e}")
