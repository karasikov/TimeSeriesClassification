#!/usr/bin/env python3
# -*- coding: utf-8 -*-


""" Signal processing procedures """


import numpy as np
import scipy


__author__ = "Mikhail Karasikov"
__copyright__ = ""
__email__ = "karasikov@phystech.edu"


class SignalLengthException(Exception):
    def __init__(self, *args, **kvargs):
        super(SignalLengthException, self).__init__(*args, **kvargs)


def ar_parameters(ts, lags):
    """Estimate AR model parameters for 1d array

    Keyword arguments:
        ts   ... (n, t) numpy.array, time-series
        lags ... list
    """
    parameters = []
    for ts_axis in np.atleast_2d(ts):
        try:
            y = ts_axis[lags[-1]:]
            X = np.ndarray((len(y), 1 + len(lags)))
            X[:, 0] = 1
            for i, lag in enumerate(lags):
                X[:, 1 + i] = ts_axis[lags[-1] - lag:-lag]
            parameters.append(np.linalg.solve(X.T.dot(X), X.T.dot(y)))
        except:
            raise SignalLengthException("Too short signal")
    return np.hstack(parameters)


def dft_features(ts, n):
    """DFT features for real time-series

    Keyword arguments:
        ts ... (n, t) numpy.array, time-series
        n  ... number of amplitudes
    """
    features = []
    for ts_axis in np.atleast_2d(ts):
        try:
            f = scipy.fft(ts_axis)
            f = f[:np.ceil((f.size + 1) / 2)]
            features.append(np.hstack((f.real[:n], f.imag[1:n])) / ts_axis.size)
        except:
            raise SignalLengthException("Too short signal")
    return np.hstack(features)
