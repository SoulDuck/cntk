#-*- coding: utf-8 -*-
import datetime
import time
import numpy as np
import os
import pandas as pd
import pickle as  pkl
pd.options.mode.chained_assignment = None  # default='warn'

from  pandas_datareader import data
#%matplotlib inline
#import cntk as C
#import cntk.tests.test_utils
#cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
#C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components




def get_stock_data(contract, s_year, s_month, s_day, e_year, e_month, e_day):
    """
    Args:
        contract (str): the name of the stock/etf
        s_year (int): start year for data
        s_month (int): start month
        s_day (int): start day
        e_year (int): end year
        e_month (int): end month
        e_day (int): end day
    Returns:
        Pandas Dataframe: Daily OHLCV bars
    """
    start = datetime.datetime(s_year, s_month, s_day)
    end = datetime.datetime(e_year, e_month, e_day)

    retry_cnt, max_num_retry = 0, 3

    while (retry_cnt < max_num_retry):
        try:
            bars = data.DataReader(contract, "google", start, end)
            return bars
        except:
            retry_cnt += 1
            time.sleep(np.random.randint(1, 10))

    print("Google Finance is not reachable")
    raise Exception('Google Finance is not reachable')




def make_stock_input():
    envvar = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'

    def is_test():
        return envvar in os.environ

    def download(data_file):
        try:
            data = get_stock_data("SPY", 2000, 1, 2, 2017, 1, 1)
        except:
            raise Exception("Data could not be downloaded")

        dir = os.path.dirname(data_file)

        if not os.path.exists(dir):
            os.makedirs(dir)

        if not os.path.isfile(data_file):
            print("Saving", data_file)
            with open(data_file, 'wb') as f:
                pkl.dump(data, f, protocol=2)
        return data

    data_file = os.path.join("data", "Stock", "stock_SPY.pkl")

    # Check for data in local cache
    if os.path.exists(data_file):
        print("File already exists", data_file)
        data = pd.read_pickle(data_file)
    else:
        # If not there we might be running in CNTK's test infrastructure
        if is_test():
            test_file = os.path.join(os.environ[envvar], 'Tutorials', 'data', 'stock', 'stock_SPY.pkl')
            if os.path.isfile(test_file):
                print("Reading data from test data directory")
                data = pd.read_pickle(test_file)
            else:
                print("Test data directory missing file", test_file)
                print("Downloading data from Google Finance")
                data = download(data_file)
        else:
            # Local cache is not present and not test env
            # download the data from Google finance and cache it in a local directory
            # Please check if there is trade data for the chosen stock symbol during this period
            data = download(data_file)

    # Feature name list
    predictor_names = []

    # Compute price difference as a feature
    data["diff"] = np.abs((data["Close"] - data["Close"].shift(1)) / data["Close"]).fillna(0)
    # 왜 가장 가장 첫번재 데이터 2016 , 11 ,14일 데이터를 Nan으로 만들지?
    # 왜 냐면 하루 전거랑 비교하기 위해서다 . 근데 처음 시작하는거는 뒤에 없으니깐 Nan이 나오는 거지
    predictor_names.append("diff")
    # Compute the volume difference as a feature
    data["v_diff"] = np.abs((data["Volume"] - data["Volume"].shift(1)) / data["Volume"]).fillna(0)
    predictor_names.append("v_diff")
    # Compute the stock being up (1) or down (0) over different day offsets compared to current dat closing price
    num_days_back = 8

    for i in range(1, num_days_back + 1):
        data["p_" + str(i)] = np.where(data["Close"] > data["Close"].shift(i), 1, 0)  # i: number of look back days
        print data["p_" + str(i)]
    # print data.head(10)
    # print data.head(-10)
    print data
    print data[-1:]

    print np.shape(data[:10])
    # If you want to save the file to your local drive
    # data.to_csv("PATH_TO_SAVE.csv")



    data["next_day"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
    data["next_day_opposite"] = np.where(data["next_day"] == 1, 0, 1)  # The label must be one-hot encoded
    # data shape ( ? , 17 )
    # Establish the start and end date of our training timeseries (picked 2000 days before the market crash)
    training_data = data["2016-11-14":"2017-10-01"]

    # We define our test data as: data["2008-01-02":]
    # This example allows to include data up to current date

    test_data = data["2017-10-01":]
    training_features = np.asarray(training_data[predictor_names], dtype="float32")
    training_labels = np.asarray(training_data[["next_day", "next_day_opposite"]], dtype="float32")

    ############################################# Make Model ###################################################
    return training_features , training_labels , test_data


if __name__ =='__main__':
    training_features, training_labels, test_data=make_stock_input()
    print np.shape(training_features)
    print np.shape(training_labels)
    print np.shape(test_data)