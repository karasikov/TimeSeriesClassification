#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Time-Series Classification"""


import os
import sys
import traceback
import collections
import pickle
import numpy as np
from sklearn import svm
import scipy.io as sio
from datetime import datetime
import ConfigParser

import sigproc
from ftpuploader import ftp_upload


__author__ = "Mikhail Karasikov"
__copyright__ = ""
__email__ = "karasikov@phystech.edu"


MASTER_ACCOUNT = "" # for unauthorized users

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

    def __init__(self, classifiers_dir, ts_data_dir, config_file):
        """ Load training dataset and classifier """

        self.config_file = config_file

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
            logging("Dataset for account <%s> is empty" % account)
            self.training_data[account] = Dataset()
        try:
            self.classifier[account] = pickle.load(open(self.classifiers_dir +
                                                        "/" + account +
                                                        "/classifier.pkl", "rb"))
        except:
            logging("No classifier was found for account <%s>; "
                    "new classifier was initialized" % account)
            self.classifier[account] = None

    def __del__(self):
        """ Retrain all classifiers in destructor """

        for account in self.classifier.keys():
            self._retrain_classifier(account)

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

                sio.savemat(mat_file, {'dataset': dataset}, do_compression=True)

                try:
                    config = ConfigParser.ConfigParser()
                    config.read(self.config_file)
                    ftp_upload(self.ts_data_dir + "/" + account, account, config)
                    logging("Data was uploaded to FTP server")
                except:
                    logging("Uploading to FTP server error:\n" + traceback.format_exc())

        except:
            logging(traceback.format_exc())

    def predict(self, account, ts_string):
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

    def train(self, account, ts_string, label):
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

            if len(self.training_data[account].y) == 0:
                logging("Empty dataset for user <%s>" % account)
                return

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
            logging("Data was dumped for user <%s>" % account)
        except:
            logging("Retraining failure:\n" + traceback.format_exc())
