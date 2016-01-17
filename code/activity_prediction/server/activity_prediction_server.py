#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Activity Prediction Server"""


import os
import socket
import argparse
import threading
import SocketServer
import re

from classification import logging, Classifier, Dataset
from Connection import Connection


__author__ = "Mikhail Karasikov"
__copyright__ = ""
__email__ = "karasikov@phystech.edu"


CLASSIFICATION_DATA_DIR = (os.path.dirname(os.path.realpath(__file__)) +
                                "/classification_data/")
TS_DATA_DIR = (os.path.dirname(os.path.realpath(__file__)) + "/ts_data/")
CONFIG_FILE = (os.path.dirname(os.path.realpath(__file__)) + "/config.ini")

PREDICT_TAG = "!PREDICT"
TRAIN_TAG = "!TRAIN"
EXIT_TAG = "!EXIT"

MASTER_ACCOUNT = "" # for unauthorized users

DELIMETER = "#$$##$$#" # don't change me!


def safe_account(name):
    return re.sub('[\W]', '', name).upper()

class Server(SocketServer.TCPServer):
    """Socket server that handles requests
    from mobile devices he communicates with.
    """

    def __init__(self, host, port):
        """Create listening socket and initialize classifier"""

        SocketServer.TCPServer.__init__(self, (host, port), TCPHandler)

        self.classifier = Classifier(CLASSIFICATION_DATA_DIR, TS_DATA_DIR, CONFIG_FILE)

    def __del__(self):
        if hasattr(self, 'classifier'):
            del self.classifier

        logging("Server has been shutted down\n")


class TCPHandler(SocketServer.BaseRequestHandler):
    """The RequestHandler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        socket, client_address = self.request, str(self.client_address[0])

        logging("New connection from " + client_address)
        try:
            connection = Connection(socket)
            received_string = connection.receive_string()
            if received_string[:len(EXIT_TAG)].upper() == EXIT_TAG:
                logging("Exit command was received. Server is shutting down...")
                threading.Thread(target=self.server.shutdown).start()
                return

            """Predict class or add new observation to training set"""
            if received_string[:len(PREDICT_TAG)].upper() == PREDICT_TAG:
                """prediction case"""
                tag, account, ts_string = received_string.split(DELIMETER)
                predicted_class = self.server.classifier.predict(safe_account(account), ts_string)
                connection.send_string(predicted_class)

            elif received_string[:len(TRAIN_TAG)].upper() == TRAIN_TAG:
                """training case"""
                tag, account, label, ts_string = received_string.split(DELIMETER)
                self.server.classifier.train(safe_account(account), ts_string, label.upper())
            else:
                logging("Unrecognized request")

        except:
            logging("Client " + client_address + " exception:")
            logging(traceback.format_exc())

        finally:
            # Clean up the connection
            logging("Connection " + client_address + " is being closed")
            socket.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='{author} <{email}>\n{copyright}'.format(
            author=__author__, email=__email__, copyright=__copyright__
        )
    )

    subparsers = parser.add_subparsers(title="mode")

    start_server = subparsers.add_parser('start')
    start_server.add_argument('-P', '--port', type=int, required=True, help='server port')
    start_server.set_defaults(func=on_start)

    stop_server = subparsers.add_parser('stop')
    stop_server.add_argument('-H', '--host', default='localhost', help='server host')
    stop_server.add_argument('-P', '--port', type=int, required=True, help='server port')
    stop_server.set_defaults(func=on_stop)

    args = parser.parse_args()
    args.func(args)


def on_start(args):
    server = Server("0.0.0.0", args.port)
    server.serve_forever()


def on_stop(args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect((args.host, args.port))

        connection = Connection(sock)
        connection.send_string(EXIT_TAG)
    except:
        sock.close()
        exit(1)


if __name__ == '__main__':
    main()
