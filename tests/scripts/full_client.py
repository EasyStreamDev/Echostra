import socket
import time
import json

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
                    "sample_rate": 44.1e3,
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

            with open("./tests/audio/morning.raw", "rb") as f:
                with open("./out.raw", "wb") as of:
                    i = 0
                    while True:
                        buffer: bytes = f.read(1024)
                        if len(buffer) == 0:
                            break
                        time.sleep(
                            11.071428571e-3
                        )  # Very approximative time to record before sending
                        send(stream_socket, buffer)

            # while True:
            time.sleep(5.0)
    except Exception as e:
        print(f"Did not work properly somewhere:\n{e}")
