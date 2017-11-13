import math
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import time

def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits np.array into training, validation and test
    """
    print 'data shape {}'.format(np.shape(data))
    pos_test = int(len(data) * (1 - test_size))
    pos_val = int(len(data[:pos_test]) * (1 - val_size))
    train, val, test = data[:pos_val], data[pos_val:pos_test], data[pos_test:]
    print 'train shape : {}'.format(np.shape(train))
    print 'val shape : {}'.format(np.shape(val))
    print 'test shape : {}'.format(np.shape(test))
    return {"train": train, "val": val, "test": test}


def generate_data(fct, x, time_steps, time_shift):
    """
    generate sequences to feed to rnn for fct(x)
    """
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(dict(a = data[0:len(data) - time_shift],
                                 b = data[time_shift:]))
    rnn_x = []
    for i in range(len(data) - time_steps + 1):
        rnn_x.append(data['a'].iloc[i: i + time_steps].as_matrix())
    rnn_x = np.array(rnn_x)

    # Reshape or rearrange the data from row to columns
    # to be compatible with the input needed by the LSTM model
    # which expects 1 float per time point in a given batch
    rnn_x = rnn_x.reshape(rnn_x.shape + (1,))
    rnn_y = data['b'].values
    rnn_y = rnn_y[time_steps - 1 :]
    rnn_y = rnn_y.reshape(rnn_y.shape + (1,))

    print 'rnn input data x shpae : {}'.format(str(np.shape(rnn_x)))
    print 'rnn input data y shpae : {}'.format(str(np.shape(rnn_y)))
    # Reshape or rearrange the data from row to columns
    # to match the input shape
    rnn_x=split_data(rnn_x)
    rnn_y = split_data(rnn_y)
    return rnn_x , rnn_y


if __name__ == '__main__':

    N = 5 # input: N subsequent values
    M = 5 # output: predict 1 value M steps ahead
    X, Y = generate_data(np.sin, np.linspace(0, 100, 10000, dtype=np.float32), N, M)

    f, a = plt.subplots(3, 1, figsize=(12, 8))
    for j, ds in enumerate(["train", "val", "test"]):
        a[j].plot(Y[ds], label=ds + ' raw');
    [i.legend() for i in a];
    plt.show(f)