import socket
import time
import json

HOST = "127.0.0.1"
PORT = 47921  # The same port as used by the server


def send(s: socket.socket, data: bytes):
    s.sendall(data)


def receive(s: socket.socket, display: bool = False) -> dict:
    response = json.loads(s.recv(2048).decode())
    if display:
        print("Server response: ", end="")
        print(json.dumps(response, indent=4))
    return response


def connect(s: socket.socket, host: str, port: int):
    s.connect((host, port))
    receive(s, display=True)


test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

connect(test_socket, HOST, PORT)

send(
    test_socket,
    json.dumps({"command": "createStream"}).encode(),
)

while 1:
    receive(test_socket, True)
    time.sleep(1.0)
