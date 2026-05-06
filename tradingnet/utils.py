"""Utility functions for data loading, processing, and visualization."""

import os
import pickle
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import requests as rq
from sklearn import preprocessing


def download_file(url, file_path):
    """Download file from Yahoo Finance."""
    r = rq.get(url)
    with open(file_path, 'wb') as f:
        f.write(r.content)


def read_file(path):
    """
    Read CSV file and cache as pickle for faster subsequent loads.

    Returns a dictionary with keys: date, open, high, low, close, adj_close, vol
    """
    pickle_filepath = path.split('.')[0] + ".pickle"

    if not os.path.exists(pickle_filepath):
        raw_data = []
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                raw_data = np.concatenate((raw_data, row))

        raw = raw_data.reshape(-1, len(row))

        with open(pickle_filepath, 'wb') as pickle_handle:
            pickle.dump(raw, pickle_handle)
    else:
        with open(pickle_filepath, 'rb') as pickle_handle:
            raw = pickle.load(pickle_handle)

    data = {
        'date': raw[1:, 0],
        'open': raw[1:, 1].astype(float),
        'high': raw[1:, 2].astype(float),
        'low': raw[1:, 3].astype(float),
        'close': raw[1:, 4].astype(float),
        'adj_close': raw[1:, 5].astype(float),
        'vol': raw[1:, 6].astype(float)
    }

    return data


def data_query(data, start_date, end_date):
    """
    Select financial data within a date range.

    Args:
        data: Dictionary of data arrays
        start_date: Start date as 'YYYY-MM-DD'
        end_date: End date as 'YYYY-MM-DD'
    """
    s_date = datetime.strptime(start_date, '%Y-%m-%d')
    e_date = datetime.strptime(end_date, '%Y-%m-%d')

    interval = e_date - s_date
    start_index = np.searchsorted(data['date'], start_date)

    for key in data:
        data[key] = data[key][start_index:start_index + interval.days + 1]

    return data


def split_timeseries(data, train, predict, step, scale=True):
    """
    Split time series data into training windows.

    Args:
        data: Array of values
        train: Number of training days
        predict: Number of prediction days
        step: Step size for sliding window
        scale: Whether to normalize data

    Returns:
        X, Y: Training and target arrays
    """
    X, Y = [], []
    for i in range(0, len(data), step):
        timeseries = np.array(data[i:i + train + predict])
        if timeseries.shape[0] == predict + train:
            if scale:
                timeseries = preprocessing.scale(timeseries)
            x_i = timeseries[:-1]
            y_i = timeseries[-1]
        else:
            break

        X.append(x_i)
        Y.append(y_i)

    return np.array(X), np.array(Y)


def split_timeseries_v2(data, train, predict, step, scale=True):
    """
    Split time series data with global normalization.

    Normalizes entire dataset before splitting, unlike v1 which normalizes per window.
    """
    X, Y = [], []
    scaled_data = preprocessing.scale(data)
    for i in range(0, len(data), step):
        timeseries = np.array(scaled_data[i:i + train + predict])
        if timeseries.shape[0] == predict + train:
            x_i = timeseries[:-1]
            y_i = timeseries[-1]
        else:
            break

        X.append(x_i)
        Y.append(y_i)

    return np.array(X), np.array(Y)


def plot_result(y_test, y_predict):
    """Plot actual (red) vs predicted (blue) values."""
    plt.plot(y_test, 'r', y_predict, 'b')
    plt.show()
