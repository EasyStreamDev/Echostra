import socket
import threading
import json

from time import sleep


HOST = "127.0.0.1"  # Localhost
PORT = 47921  # Port number


class _ThreadSafeDict:
    def __init__(self, val: dict = {}):
        self._dict = val
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

        # Bind the socket to a specific host and port
        self._server_socket.bind((host, port))
        # Set the timeout for accepting client connections
        self._server_socket.settimeout(50e-3)  # Timeout set to 50ms

    # Server main function (blocking)
    def run(self) -> None:
        # Listen for incoming connections
        self._server_socket.listen()
        self._running = True

        print(f"TCP Server listening on {HOST}:{PORT}")

        # Accept incoming connections in the main thread
        while self._running:
            try:
                # Accept client connection
                client_socket, client_address = self._server_socket.accept()
                # Handle the client connection in a separate thread
                client_thread = threading.Thread(
                    target=self._handle_client_connect,
                    args=(client_socket, client_address),
                )
                # Add client and related thread to client list
                self._clients[client_address] = {
                    "socket": client_socket,
                    "thread": client_thread,
                    "exit_event": threading.Event(),
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
            self._clients.get(key)["exit_event"].set()
        # Waiting for all clients thread to terminate.
        while self._clients.items().__len__() != 0:
            self._remove_closed_connections()
            sleep(50e-3)  # Sleep for 50ms

    # Function to handle client connections
    def _handle_client_connect(self, client_socket: socket.socket, client_address: str):
        print(f"Connected to client: {client_address}")
        exit_event: threading.Event = self._clients.get(client_address)["exit_event"]

        client_socket.sendall(TCPServer.CONNECTION_SUCCESS_MSG)
        # Receive and process client data
        while not exit_event.is_set():
            data = client_socket.recv(1024)
            if not data:
                break
            # Process the received data
            print(f"Received data from {client_address}: {data.decode()}")

            # Echo the data back to the client
            client_socket.sendall(data)

        # Close the connection
        client_socket.close()
        print(f"Connection closed with client: {client_address}")

    # Function to remove closed client connection
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
            if value["socket"].fileno() != -1 and value["thread"].is_alive():
                staying_clients[key] = value
            else:
                disconnected_clients[key] = value
                value["exit_event"].set()

        self._clients = staying_clients

        for key, value in disconnected_clients.items():
            value["thread"].join()


server = TCPServer(host=HOST, port=PORT)

server.run()
