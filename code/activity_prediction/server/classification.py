#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Time-Series Classification"""


import os
import sys
import traceback
import collections
import pickle
import numpy as np
import scipy.io as sio
from datetime import datetime
from contextlib import redirect_stdout

from sklearn import svm
import sklearn.preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn import svm, cross_validation
from sklearn.pipeline import Pipeline

import timeseries
import feature_extraction
import multiclass


__author__ = "Mikhail Karasikov"
__copyright__ = ""
__email__ = "karasikov@phystech.edu"


MASTER_ACCOUNT = ""  # for unauthorized users

TS_FREQUENCY = 20


def logging(message):
    sys.stderr.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S, ") + message + "\n")


class Classifier:
    """ Machine learning time-series classifier """

    def __init__(self, classifiers_dir, ts_data_dir, retraining_delay, upload_data):
        """ Load training dataset and classifier """

        self.upload_data = upload_data

        self.classifiers_dir = classifiers_dir
        self.ts_data_dir = ts_data_dir
        self.retraining_delay = retraining_delay

        self.classifier = {}
        self._load_classifier(MASTER_ACCOUNT)

    def _load_classifier(self, account):
        try:
            with open(self.classifiers_dir + "/" +
                      account + "/classifier.pkl", "rb") as file:
                self.classifier[account] = pickle.load(file)
        except:
            logging("No classifier was found for account <%s>; "
                    "new classifier was initialized" % account)
            self.classifier[account] = None

    def _save_ts(self, ts, label, account):
        """Save new time-series"""

        try:
            data_path = self.ts_data_dir + "/" + account

            try:
                dataset = timeseries.TSDataset.load_from_mat(data_path + "/dataset.mat")
            except:
                dataset = timeseries.TSDataset()
            dataset.add_observation(ts, label)

            if not os.path.exists(data_path):
                os.makedirs(data_path)

            np.savetxt(data_path + "/last_ts.csv", ts.T,
                       delimiter=",", comments="", header="t,x,y,z")
            dataset.save_to_mat(data_path + "/dataset.mat", do_compression=True)

            self.upload_data(data_path, account)

            return len(dataset.label) * (len(set(dataset.label) - {"?"}) > 1)
        except:
            logging(traceback.format_exc())
            return -1

    def predict(self, account, ts):
        """ Predict class' label of the time-series """

        self._save_ts(ts, "?", account)
        try:
            features = get_features([ts])
        except feature_extraction.SignalLengthException as e:
            return str(e)
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

    def train(self, account, ts, label):
        """ Adding new observation and to the training set """

        training_set_size = self._save_ts(ts, label, account)
        if training_set_size > 0 and training_set_size % self.retraining_delay == 0:
            self._retrain_classifier(account)

    def _retrain_classifier(self, account):
        """ Fit classifier with stored training data """

        try:
            logging("Classifier retraining...")

            try:
                dataset = timeseries.TSDataset.load_from_mat(self.ts_data_dir + "/" + account + "/dataset.mat")
            except:
                logging("Empty dataset for user <%s>" % account)
                return

            observations = dataset.label != "?"
            X = get_features(dataset.ts[observations])
            scaler = sklearn.preprocessing.MinMaxScaler((-1, 1))
            X_normalized = scaler.fit_transform(X)
            y = dataset.label[observations]

            data_path = self.ts_data_dir + "/" + account + "/"

            with open(data_path + 'retraining_out.txt', 'w') as f:
                with redirect_stdout(f):
                    try:
                        grid_search_cv = GridSearchCV(
                            svm.SVC(), {
                                'gamma': np.logspace(-4, 0, 10),
                                'C': np.logspace(0, 3, 10),
                            },
                            scoring=None,
                            n_jobs=1,
                            refit=False,
                            cv=cross_validation.StratifiedKFold(
                                y, 5, shuffle=True, random_state=17
                            ),
                            verbose=1,
                            error_score='raise'
                        ).fit(X_normalized, y)

                        multiclass.plot_grid_search_scores(
                            grid_search_cv,
                            fig_name=data_path + "cross_validation.jpg",
                            show_plot=False
                        )

                        confusion_mean = multiclass.cross_val_score(
                            svm.SVC(**grid_search_cv.best_params_),
                            X_normalized, y,
                            cross_validation.StratifiedKFold(
                                y, 10, shuffle=True, random_state=18
                            ),
                            fig_name=data_path + "classification_accuracy.jpg",
                            show_plot=False
                        )
                    except Exception as e:
                        logging("Fitting classifier failure: " + str(e))
                        return

            self.classifier[account] = Pipeline([
                ('scaler', scaler),
                ('svc', svm.SVC(**grid_search_cv.best_params_))
            ])
            self.classifier[account].fit(X, y)

            logging("Ok")

            account_folder = self.classifiers_dir + "/" + account + "/"
            if not os.path.exists(account_folder):
                os.makedirs(account_folder)

            pickle.dump(self.classifier[account],
                        open(account_folder + "/classifier.pkl", "wb"))
            logging("Classifier was dumped for user <%s>" % account)
        except:
            logging("Retraining failure:\n" + traceback.format_exc())


def get_features(ts_dataset):
    ts_smoothed = [
        timeseries.transform_frequency(ts[0], ts[1:], TS_FREQUENCY, kind='linear')[1]
        for ts in ts_dataset
    ]

    def histogram(ts):
        return np.histogram(ts, density=True, bins=10)[0] * (ts.max() - ts.min()) / 10

    X = timeseries.ExtractFeatures(ts_smoothed,
        lambda ts: ts.mean(1),
        lambda ts: ts.std(1),
        lambda ts: np.abs(ts - ts.mean(1).reshape(-1, 1)).mean(1),
        lambda ts: np.sqrt((ts ** 2).sum(0)).mean(),
        lambda ts: histogram(ts[0]),
        lambda ts: histogram(ts[1]),
        lambda ts: histogram(ts[2]),
        lambda ts: feature_extraction.ar_parameters(ts, np.arange(1, 5)),
        lambda ts: feature_extraction.dft_features(np.sqrt((ts[:3] ** 2).sum(0)), 5)[1:]
    )

    return X
