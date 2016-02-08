#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" Time-Series package """


import os
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d


__author__ = "Mikhail Karasikov"
__copyright__ = ""
__email__ = "karasikov@phystech.edu"


class TSDataset(object):
    """Dataset class

    Attributes:
        ts ... (m,) numpy.array, time-series
        label ... (m,) numpy.array, class labels
    """

    def __init__(self, ts=np.array([]), label=[]):
        self.__dataset = np.array((ts, label), dtype=[('ts', 'O'), ('label', 'O')])

    def __getattr__(self, attr):
        if attr == 'ts':
            return self.__dataset['ts'].ravel()
        elif attr == 'label':
            return np.hstack(self.__dataset['label'].ravel()).ravel()
        else:
            raise AttributeError("'TSDataset' object has no attribute '%s'" % attr)

    def __getitem__(self, i):
        return (self.ts[i], self.label[i])

    def __len__(self):
        return self.ts.size

    def load_from_mat(self, mat_file):
        self.__dataset = sio.loadmat(mat_file)['dataset']

    def save_to_mat(self, mat_file, do_compression=True):
        if not os.path.exists(os.path.dirname(mat_file)):
            os.makedirs(os.path.dirname(mat_file))

        sio.savemat(mat_file,
                    {'dataset': self.__dataset},
                    do_compression=do_compression)

    def add_observation(self, ts, label):
        self.extend(np.array((ts, label), dtype=[('ts', 'O'), ('label', 'O')]))

    def extend(self, new_observations):
        self.__dataset = np.vstack((self.__dataset, new_observations))


def ExtractFeatures(ts, *extractors):
    """Extract features from a given set of time-series

    Keyword arguments:
        ts ... (m,) numpy.array, set of time-series
        extractors ... callable, extract features from a single time-seres
    """
    if len(extractors) is 0:
        raise ValueError("Feature extractor was not specified")

    def get_features(ts):
        return np.hstack(map(lambda extractor: extractor(ts), extractors))
    return np.vstack(map(get_features, ts))


def transform_frequency(time, ts, freq, kind='linear'):
    """1-D cubic spline interpolation

    Keyword arguments:
        time ... (t,) numpy.array, time points
        ts   ... (n, t) numpy.array, time-series
        freq ... frequency
        kind ... interpolation method: 'linear', 'quadratic', 'cubic'
    """
    t = np.linspace(time[0], time[-1], (time[-1] - time[0]) * freq + 1, endpoint=True)
    ts_transformed = []
    for ts_axis in np.atleast_2d(ts):
        interpolation = interp1d(time, ts_axis.ravel(), kind)
        ts_transformed.append(interpolation(t))
    return t, np.vstack(ts_transformed)
