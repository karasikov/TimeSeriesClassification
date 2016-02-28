#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Activity Prediction Server"""


import os
import socket
import argparse
import traceback
import threading
import socketserver
import configparser
import re
import numpy as np

from connection import Connection
from ftpuploader import ftp_upload_dir
from classification import logging, Classifier


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

MASTER_ACCOUNT = ""  # for unauthorized users

DELIMETER = "#$$##$$#"  # don't change me!


def safe_account(name):
    return re.sub('[\W]', '', name).upper()


def str_to_ts(ts_string):
    """Extract time-series from string to numpy array"""

    ts = np.matrix(ts_string.replace("\n", ";"), np.float64)
    time = np.array(ts[:, 0], np.float64).ravel() / 1000  # ms to seconds
    time = time - time[0]
    ts = np.array(ts[:, 1:]).T
    return np.vstack((time, ts))


def upload_data(data_path, destination_path):
    try:
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)
        ftp_upload_dir(data_path, destination_path, config)
        logging("Data was uploaded to FTP server")
    except:
        # logging("Uploading to FTP server error:\n" + traceback.format_exc())
        logging("Uploading to FTP server error")


class Server(socketserver.TCPServer):
    """Socket server that handles requests
    from mobile devices he communicates with.
    """

    def __init__(self, host, port):
        """Create listening socket and initialize classifier"""

        socketserver.TCPServer.__init__(self, (host, port), TCPHandler)

        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)
        retraining_delay = int(config.get('Classification', 'retraining_delay'))
        self.classifier = Classifier(CLASSIFICATION_DATA_DIR, TS_DATA_DIR,
                                     retraining_delay=retraining_delay,
                                     upload_data=upload_data)

    def __del__(self):
        if hasattr(self, 'classifier'):
            del self.classifier

        logging("Server has been shutted down\n")


class TCPHandler(socketserver.BaseRequestHandler):
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
                logging("Exit command was received. Server is being shut down...")
                threading.Thread(target=self.server.shutdown).start()
                return

            """Predict class or add new observation to training set"""

            tag, account, *label, ts_string = received_string.split(DELIMETER)

            if tag.upper() == PREDICT_TAG:
                """prediction case"""
                predicted_class = self.server.classifier.predict(
                    safe_account(account),
                    str_to_ts(ts_string)
                )
                connection.send_string(predicted_class)

            elif tag.upper() == TRAIN_TAG:
                """training case"""
                self.server.classifier.train(
                    safe_account(account),
                    str_to_ts(ts_string),
                    label[0].upper()
                )
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

    subparsers = parser.add_subparsers(dest="mode")
    subparsers.required = True

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
