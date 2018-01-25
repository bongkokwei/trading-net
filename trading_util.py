import requests as rq
import csv
import numpy as np
from sklearn import preprocessing
from datetime import datetime
import os
import pickle
import matplotlib.pyplot as plt

def download_file(url, file_path):
    """
    Downloads file from yahoo fiance
    Bugs: getting unauthorised error
    """

    r = rq.get(url)
    with open(file_path, 'wb') as f:
        f.write(r.content)

def read_file(path):
    """
    Check if file has been read before then
    Reads csv file and import them into python dictionary
    need more work: what if I need to read a new file(DONE)
    pickle file will have same name as csv file.
    """
    pickle_filepath = path.split('.')[0]+".pickle"

    if not os.path.exists(pickle_filepath):
        # Read data set from disk
        raw_data = []
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter = ',')
            for row in reader:
                raw_data = np.concatenate((raw_data,row))
        f.close()
        # reshapes array to [no. of entries, num_col]
        raw = raw_data.reshape(-1, len(row))

        with open(pickle_filepath, 'wb') as pickle_handle:
            pickle.dump(raw, pickle_handle)
    else:
        with open(pickle_filepath, 'rb') as pickle_handle:
            raw = pickle.load(pickle_handle)

    data = {'date': raw[1:,0],
            'open': raw[1:,1].astype(float),
            'high': raw[1:,2].astype(float),
            'low': raw[1:,3].astype(float),
            'close': raw[1:,4].astype(float),
            'adj_close': raw[1:,5].astype(float),
            'vol': raw[1:,6].astype(float)}

    return data

def data_query(data, start_date, end_date):
    """
    select financial data based on dates
    date format: 'YYYY-MM-DD'
    Assume start_date always earlier than end_date
    """
    # Convert string dates to datetime objects
    # Placeholder to validate start_date < end_date
    s_date = datetime.strptime(start_date, '%Y-%m-%d')
    e_date   = datetime.strptime(end_date  , '%Y-%m-%d')

    interval = e_date-s_date
    start_index = np.searchsorted(data['date'], start_date)

    for key in data:
        data[key] = data[key][start_index:start_index+interval.days+1]

    return data

def split_timeseries(data, train, predict, step, scale=True):
    """
    split time series data into:
    X (num of 'train' days)
    Y (train + predict days)
    """
    X, Y = [], []
    for i in range(0, len(data), step):
        timeseries = np.array(data[i:i+train+predict])
        if timeseries.shape[0] == predict+train:
            if scale: timeseries = preprocessing.scale(timeseries)
            x_i = timeseries[:-1] #from first to second last element
            y_i = timeseries[-1]  #last element
        else:
            break

        X.append(x_i)
        Y.append(y_i)

    return np.array(X), np.array(Y)

def split_timeseries_v2(data, train, predict, step, scale=True):
    """
    split time series data into:
    X (num of 'train' days)
    Y (train + predict days)
    this will normalise ALL data instead of just sliding window
    """
    X, Y = [], []
    scaled_data = preprocessing.scale(data)
    for i in range(0, len(data), step):
        timeseries = np.array(scaled_data[i:i+train+predict])
        if timeseries.shape[0] == predict+train:
            x_i = timeseries[:-1] #from first to second last element
            y_i = timeseries[-1]  #last element
        else:
            break

        X.append(x_i)
        Y.append(y_i)

    return np.array(X), np.array(Y)

def plot_result(y_test, y_predict):
    plt.plot(y_test, 'r', y_predict, 'b')
    plt.show
    pass
