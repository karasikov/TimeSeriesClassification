#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

""" Signal processing procedures """

class TooShortSignalException(Exception):
    def __init__(self, *args, **kvargs):
        super(TooShortSignalException, self).__init__(*args, **kvargs)

def binary_search(data, val):
    lo, hi = 0, len(data) - 1
    best_ind = lo
    while lo <= hi:
        mid = lo + (hi - lo) / 2
        if data[mid] < val:
            lo = mid + 1
        elif data[mid] > val:
            hi = mid - 1
        else:
            best_ind = mid
            break
        # check if data[mid] is closer to val than data[best_ind] 
        if abs(data[mid] - val) < abs(data[best_ind] - val):
            best_ind = mid
    return best_ind

def transform_frequency(time, ts, freq):
    """ Extrapolation to the mesh """

    if len(time) < 2:
        raise TooShortSignalException("Too short signal")

    def exp_weight(x, m, sigma):
        return np.exp(-(x - m) ** 2 / (2 * sigma ** 2))

    T = float(1) / freq
    NUM_NEIGHBOURS = 2
    SIGMA = NUM_NEIGHBOURS * T / 2
    n = int(float(time[-1] - time[0]) / T)
    ts = np.matrix(ts)
    ts_transformed = np.matrix(np.zeros((n, ts.shape[1])))
    time_mesh = time[0] + np.reshape(np.array(range(0, n)) * T, (-1, 1))

    for i in range(0, n):
        sum_weight = 0
        t = time_mesh[i]
        nearest = binary_search(time, t)

        for j in range(max(0, nearest - NUM_NEIGHBOURS),
                       min(len(time), nearest + NUM_NEIGHBOURS)):
            weight = exp_weight(time[j], t, SIGMA)
            sum_weight += weight
            ts_transformed[i, :] += weight * ts[j, :]
        ts_transformed[i, :] /= sum_weight
    return (time_mesh, np.array(ts_transformed))

def get_features(ts):
    """ This function defines feature-space (design matrix) """

    features = []
    features.extend(ts.mean(0))
    features.extend(ts.std(0))
    features.extend(np.abs(ts - ts.mean(0)).mean(0))

    features.extend([((ts ** 2).sum(1) ** 0.5).mean()])

    features.extend(calculate_ar_coefficients(ts[:, 0], [1, 3, 5, 7, 9, 11, 13, 15]))
    features.extend(calculate_ar_coefficients(ts[:, 1], [1, 3, 5, 7, 9, 11, 13, 15]))
    features.extend(calculate_ar_coefficients(ts[:, 2], [1, 3, 5, 7, 9, 11, 13, 15]))

    return np.array(features)

def calculate_ar_coefficients(ts, lags):
    """ Calculate AR model coefficients for 1d array """

    try:
        ts = np.matrix(ts).reshape((-1, 1))
        y = ts[lags[-1] + 1:, 0]
        X = np.matrix([1] * len(y)).reshape((-1, 1))
        for lag in lags:
            X = np.hstack((X, ts[lags[-1] + 1 - lag:-lag, 0]))

        return np.linalg.inv(X.T * X) * (X.T * y)
    except:
        raise TooShortSignalException("Too short signal")
