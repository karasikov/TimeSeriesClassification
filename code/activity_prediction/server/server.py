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
import sigproc
import scipy.io as sio
from datetime import datetime

""" Activity Recognition server """

CLASSIFICATION_DATA_DIR = (os.path.dirname(os.path.realpath(__file__)) +
                                "/classification_data/")
TS_DATA_DIR = (os.path.dirname(os.path.realpath(__file__)) +
                                "/ts_data/")

PREDICT_TAG = "!PREDICT"
TRAIN_TAG = "!TRAIN"
EXIT_TAG = "!EXIT"

MASTER_ACCOUNT = "" # for unauthorized users

DELIMETER = "#$$##$$#" # don't change me!

RETRAINING_DELAY = 5
FREQUENCY = 20

def logging(message):
    sys.stderr.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S, ") + message + "\n")

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

    def __init__(self, classifiers_dir, ts_data_dir):
        """ Load training dataset and classifier """

        self.classifiers_dir = classifiers_dir
        self.ts_data_dir = ts_data_dir

        self.classifier = {}
        self.training_data = {}
        self._load_classifier(MASTER_ACCOUNT)

    def _load_classifier(self, account):
        try:
            self.training_data[account] = pickle.load(open(self.classifiers_dir +
                                                           "/" + account +
                                                           "/training_data.pkl", "rb"))
        except:
            logging("Dataset for account " + account + " is empty...")
            self.training_data[account] = Dataset()
        try:
            self.classifier[account] = pickle.load(open(self.classifiers_dir +
                                                        "/" + account +
                                                        "/classifier.pkl", "rb"))
        except:
            logging("No classifier found for account " + account +
                    ". New classifier initialization...")
            self.classifier[account] = None

    def __del__(self):
        """ Retrain all classifiers in destructor """

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
            logging("Unrecognized request was received")

    def _prepare_ts(self, ts_string):
        """ Extract time-series from string to numpy array """

        ts = np.matrix(ts_string.replace("\n", ";").encode("utf-8"), np.float64)
        time = np.array(ts[:, 0], np.float64) / 1000 # ms to seconds
        time = time - time[0]
        ts = np.array(ts[:, 1:])

        time, ts = sigproc.transform_frequency(time, ts, FREQUENCY)
        return (time, ts)

    def _ts_logging(self, time, ts, label, account):
        """ Logging time-series """

        try:
            if len(self.ts_data_dir) > 0:
                mat_file = self.ts_data_dir + "/" + account + "/dataset.mat"
                new_ts = np.array((np.hstack((time, ts)).T, label),
                                  dtype=[('ts', 'O'), ('label', 'O')])
                try:
                    dataset = sio.loadmat(mat_file)['dataset']
                    dataset = np.vstack((dataset.T, new_ts)).T
                except:
                    dataset = new_ts

                if not os.path.exists(os.path.dirname(mat_file)):
                    os.makedirs(os.path.dirname(mat_file))

                sio.savemat(mat_file, {'dataset': dataset})
        except:
            logging(traceback.format_exc())

    def _predict(self, account, ts_string):
        """ Predict class' label of the time-series """

        try:
            time, ts = self._prepare_ts(ts_string)
            self._ts_logging(time, ts, "?", account)
            features = sigproc.get_features(np.array(ts))
        except sigproc.TooShortSignalException as e:
            return "Too short signal"
        except:
            logging(traceback.format_exc())
            return "Bad signal"

        if account not in self.classifier.keys():
            self._load_classifier(account)

        if self.classifier[account] is None:
            return "Train classifier"

        try:
            return self.classifier[account].predict(features)[0]
        except:
            return "Add samples"

    def _train(self, account, ts_string, label):
        """ Adding new observation and to the training set """

        time, ts = self._prepare_ts(ts_string)
        self._ts_logging(time, ts, label, account)
        features = sigproc.get_features(ts)

        if account not in self.training_data.keys():
            self._load_classifier(account)

        try:
            self.training_data[account].X = np.vstack([self.training_data[account].X, features])
            self.training_data[account].y.append(label)
        except:
            self.training_data[account].X = np.array([features])
            self.training_data[account].y = [label]
        if len(self.training_data[account].y) % RETRAINING_DELAY == 0:
            self._retrain_classifier(account)

    def _retrain_classifier(self, account):
        """ Fit classifier with stored training data """

        try:
            logging("Classifier retraining...")
            self.classifier[account] = svm.SVC(C=10, gamma=0.1)
            self.classifier[account].fit(self.training_data[account].X, self.training_data[account].y)
            logging("Ok")

            account_folder = self.classifiers_dir + "/" + account + "/"
            if not os.path.exists(os.path.dirname(account_folder)):
                os.makedirs(os.path.dirname(account_folder))

            pickle.dump(self.classifier[account],
                        open(account_folder + "/classifier.pkl", "wb"))
            pickle.dump(self.training_data[account],
                        open(account_folder + "/training_data.pkl", "wb"))
            logging("Data was dumped for user {:s}".format(account))
        except:
            logging("Retraining failure:\n" + traceback.format_exc())

class Server:
    """ Server which communicates with mobile devices and handles requests """

    def __init__(self, host, port):
        """ Create listening socket and initialize classifier """

        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the port
        server_address = (host, port)
        self.sock.bind(server_address)
        self.classifier = Classifier(CLASSIFICATION_DATA_DIR, TS_DATA_DIR)

    def run(self):
        """ Start listening """

        # Listen for incoming connections
        self.sock.listen(1)

        while True:
            logging("waiting for a connection...")
            client_socket, client_address = self.sock.accept()

            # There server has new connection
            # and can interact with client through blocking TCP socket |client_socket|
            logging("connection from " + str(client_address))
            try:
                connection = Connection(client_socket)
                received_string = connection.receive_string()
                if received_string[:len(EXIT_TAG)].upper() == EXIT_TAG:
                    logging("exit command was received, server will be halted")
                    self.sock.close()
                    return

                # Predict class or add new observation to training set
                result = self.classifier.handle(received_string)
                if result is not None:
                    # Send predicted class
                    connection.send_string(result)
            except:
                logging("client " + str(client_address) + " exception:")
                logging(traceback.format_exc())
            finally:
                # Clean up the connection
                logging("connection " + str(client_address) + " is being closed")
                client_socket.close()


def main():
    if len(sys.argv) != 2:
        print >> sys.stderr, "Usage: {:s} <port>".format(sys.argv[0])
        exit(1)

    server = Server("0.0.0.0", int(sys.argv[1]))
    server.run()


main()
