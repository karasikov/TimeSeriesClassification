#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import sys
import os
import traceback
import collections
import pickle
import numpy as np
from sklearn import svm
from Connection import Connection

""" Activity Recognition server """

CLASSIFICATION_DATA_FOLDER = (os.path.dirname(os.path.realpath(__file__)) +
                                "/classification_data")

PREDICT_TAG = "!PREDICT"
TRAIN_TAG = "!TRAIN"
EXIT_TAG = "!EXIT"

MASTER_ACCOUNT = "" # for unauthorized users

DELIMETER = "#$$##$$#" # don't change it!

class Dataset:
    """ Dataset class
        fields: X --- design-matrix
                y --- classes' labels
    """
    def __init__(self):
        self.X = np.array([])
        self.y = []

class Classifier:
    """ Machine learning time-series classifier """

    def __init__(self, path):
        """ Load training dataset and classifier """

        self.PATH_TO_DUMP = path

        self.classifier = {}
        self.training_data = {}
        try:
            self.training_data[MASTER_ACCOUNT] = pickle.load(open(self.PATH_TO_DUMP +
                                                                  "/" + MASTER_ACCOUNT +
                                                                  "/training_data.pkl", "rb"))
        except:
            sys.stderr.write("ERROR: Dataset is empty...\n")
            self.training_data[MASTER_ACCOUNT] = Dataset()
        try:
            self.classifier[MASTER_ACCOUNT] = pickle.load(open(self.PATH_TO_DUMP +
                                                               "/" + MASTER_ACCOUNT +
                                                               "/classifier.pkl", "rb"))
        except:
            sys.stderr.write("ERROR: No classifier found in the folder! New classifier initialization...\n")
            self.classifier[MASTER_ACCOUNT] = None

    def __del__(self):
        # retrain classifier on exit
        for account in self.classifier.keys():
            self._retrain_classifier(account)


    def handle(self, request):
        """ Handle request: prediction/training """

        #print request
        if request[:len(PREDICT_TAG)].upper() == PREDICT_TAG:
            # prediction case
            tag, account, ts_string = request.split(DELIMETER)
            return self._predict(account.upper(), ts_string)
        elif request[:len(TRAIN_TAG)].upper() == TRAIN_TAG:
            # training case
            tag, account, label, ts_string = request.split(DELIMETER)
            self._train(account.upper(), ts_string, label.upper())
        else:
            pass

    def _prepare_ts(self, ts_string):
        """ extract time-series from string to numpy array """

        ts = np.matrix(ts_string.replace("\n", ";").encode("utf-8"), np.float64)
        time = np.array(ts[:, 0], np.uint64)
        ts = np.array(ts[:, 1:])
        return (time, ts)

    def _predict(self, account, ts_string):
        """ Predict class' label of the time-series """

        if account not in self.classifier.keys() or self.classifier[account] is None:
            return "Train classifier"
        else:
            time, ts = self._prepare_ts(ts_string)
            features = self._get_features(np.array(ts))

            #print features
            return self.classifier[account].predict(features)[0]

    def _train(self, account, ts_string, label):
        """ Adding new observation and to the training set """

        time, ts = self._prepare_ts(ts_string)
        features = self._get_features(ts)

        # Check whether this user new
        if account not in self.training_data.keys():
            self.training_data[account] = Dataset()
            self.classifier[account] = None

        try:
            self.training_data[account].X = np.vstack([self.training_data[account].X, features])
            self.training_data[account].y.append(label)
        except:
            self.training_data[account].X = features
            self.training_data[account].y = [label]
        if len(self.training_data[account].X) % 5 == 0:
            self._retrain_classifier(account)

    def _retrain_classifier(self, account):
        """ Fit classifier with stored training data """

        sys.stderr.write("Classifier retraining...\n")
        self.classifier[account] = svm.SVC(C=10, gamma=0.1)
        self.classifier[account].fit(self.training_data[account].X, self.training_data[account].y)
        sys.stderr.write("Ok\n")

        account_folder = self.PATH_TO_DUMP + "/" + account + "/"
        if not os.path.exists(os.path.dirname(account_folder)):
            os.makedirs(os.path.dirname(account_folder))

        pickle.dump(self.classifier[account],
                    open(account_folder + "/classifier.pkl", "wb"))
        pickle.dump(self.training_data[account],
                    open(account_folder + "/training_data.pkl", "wb"))
        sys.stderr.write("Data was dumped for user {:s}\n".format(account))

    def _get_features(self, ts):
        """ This function defines feature-space (design matrix) """

        features = []
        features.extend(ts.mean(0))
        features.extend(ts.std(0))
        features.extend(np.abs(ts - ts.mean(0)).mean(0))
        features.extend([((ts ** 2).sum(1) ** 0.5).mean()])
        return np.array(features)

class Server:
    """ Server which communicates with mobile devices and handles requests """

    def __init__(self, host, port):
        """ Create listening socket and initialize classifier """

        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the port
        server_address = (host, port)
        self.sock.bind(server_address)
        self.classifier = Classifier(CLASSIFICATION_DATA_FOLDER)

    def run(self):
        """ Start listening """

        # Listen for incoming connections
        self.sock.listen(1)

        while True:
            sys.stderr.write('waiting for a connection...\n')
            client_socket, client_address = self.sock.accept()

            # There server has new connection
            # and can interact with client through blocking TCP socket |client_socket|
            print >> sys.stderr, 'connection from', client_address
            try:
                connection = Connection(client_socket)
                received_string = connection.receive_string()
                if received_string[:len(EXIT_TAG)].upper() == EXIT_TAG:
                    print >> sys.stderr, "exit command was received, server will be halted"
                    self.sock.close()
                    return

                # Predict class or add new observation to training set
                result = self.classifier.handle(received_string)
                if result is not None:
                    # Send predicted class
                    connection.send_string(result)
            except:
                print >> sys.stderr, "client ", client_address, " exception:"
                print >> sys.stderr, traceback.format_exc()
            finally:
                # Clean up the connection
                print >> sys.stderr, "connection", client_address, "is being closed"
                client_socket.close()


def main():
    if len(sys.argv) != 2:
        print >> sys.stderr, "Usage: {:s} <port>".format(sys.argv[0])
        exit(1)

    server = Server("0.0.0.0", int(sys.argv[1]))
    server.run()


main()
