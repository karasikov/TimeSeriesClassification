#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""TCP Connection Interface"""


import socket
import sys
import struct


MAX_STRING_SIZE = 10 ** 8


class Connection:
    """ Class for communication through socket
    It guarantees correct transmission for TCP socket
    """

    def __init__(self, socket):
        self.socket = socket

    def send_string(self, string):
        string_length = len(string)
        self.socket.sendall(struct.pack('!L', string_length) +
                            string.encode('utf-8'))

    def receive_string(self):
        data = b""
        while len(data) < 4:
            data += self.socket.recv(4 - len(data))

        string_length = struct.unpack('!L', data)[0]

        if string_length > MAX_STRING_SIZE:
            raise IOError("Received string length longer than maximum allowed" +
                            " (" + str(string_length) + " > " + str(MAX_STRING_SIZE) + ")")
        # Receive data by small chunks and reconstruct the string
        received = []
        while string_length > 0:
            received.append(self.socket.recv(min(16, string_length)))
            string_length -= len(received[-1])

        return b"".join(received).decode("utf-8")
