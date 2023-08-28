from src.TCPServer import TCPServer

HOST = "127.0.0.1"  # Localhost
PORT = 47921  # Port number

server = TCPServer(host=HOST, port=PORT)
server.run()
