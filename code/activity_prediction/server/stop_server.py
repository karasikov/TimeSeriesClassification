import socket
import sys
from Connection import Connection

""" To send exit code to local server """

EXIT_CODE = "!exit:"

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', int(sys.argv[1])))

connection = Connection(sock)
connection.send_string(EXIT_CODE)
