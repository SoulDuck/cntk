import os
import pandas as pd
import numpy as np
from urllib import urlretrieve
def generate_solar_data(input_url, time_steps, normalize=1, val_size=0.1, test_size=0.1):
    """
    generate sequences to feed to rnn based on data frame with solar panel data
    the csv has the format: time ,solar.current, solar.total
     (solar.current is the current output in Watt, solar.total is the total production
      for the day so far in Watt hours)
    """
    # try to find the data file local. If it doesn't exists download it.
    cache_path = os.path.join("data", "iot")
    cache_file = os.path.join(cache_path, "solar.csv")
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    if not os.path.exists(cache_file):
        urlretrieve(input_url, cache_file)
        print("downloaded data successfully from ", input_url)
    else:
        print("using cache for ", input_url)

    df = pd.read_csv(cache_file, index_col="time", parse_dates=['time'], dtype=np.float32)


    df["date"] = df.index.date

    # normalize data
    df['solar.current'] /= normalize
    df['solar.total'] /= normalize
    # group by day, find the max for a day and add a new column .max
    grouped = df.groupby(df.index.date).max()
    print df[:1]
    print np.shape(df)
    grouped.columns = ["solar.current.max", "solar.total.max", "date"]# change grouped columns name
    print grouped[:1]
    print np.shape(grouped)
    # merge continuous readings and daily max values into a single frame
    df_merged = pd.merge(df, grouped, right_index=True, on="date")
    df_merged = df_merged[["solar.current", "solar.total",
                           "solar.current.max", "solar.total.max"]]
    print df_merged[:1]
    print np.shape(df_merged)

    # we group by day so we can process a day at a time.
    grouped = df_merged.groupby(df_merged.index.date)
    per_day = []
    for _, group in grouped:
        per_day.append(group)

    # split the dataset into train, validatation and test sets on day boundaries
    val_size = int(len(per_day) * val_size)
    print 'validation  size {}'.format(val_size)
    test_size = int(len(per_day) * test_size)
    print 'test size {}'.format(test_size)
    next_val = 0
    next_test = 0

    result_x = {"train": [], "val": [], "test": []}
    result_y = {"train": [], "val": [], "test": []}

    # generate sequences a day at a time
    for i, day in enumerate(per_day):


        # if we have less than 8 datapoints for a day we skip over the
        # day assuming something is missing in the raw data
        total = day["solar.total"].values
        if len(total) < 8:
            continue
        if i >= next_val:
            current_set = "val"
            next_val = i + int(len(per_day) / val_size)

        elif i >= next_test:
            current_set = "test"
            next_test = i + int(len(per_day) / test_size)
        else:
            current_set = "train"
        max_total_for_day = np.array(day["solar.total.max"].values[0])
        #print 'total : {}'.format(len(total))
        for j in range(2, len(total)):
            #print 'current set : {}'.format(current_set)
            result_x[current_set].append(total[0:j])
            result_y[current_set].append([max_total_for_day])
            if j >= time_steps:
                break


    # make result_y a numpy array
    for ds in ["train", "val", "test"]:
        result_y[ds] = np.array(result_y[ds])

    print 'n train x : ',np.shape(result_x['train'])
    print 'n test  x  : ',np.shape(result_x['val'])
    print 'n valication x : ',np.shape(result_x['test'])
    print 'n train y: ',np.shape(result_y['train'])
    print 'n test  y  : ',np.shape(result_y['val'])
    print 'n validation  y  : ',np.shape(result_y['test'])


    return result_x, result_y

# We keep upto 14 inputs from a day
TIMESTEPS = 14

# 20000 is the maximum total output in our dataset. We normalize all values with
# this so our inputs are between 0.0 and 1.0 range.
NORMALIZE = 20000

X, Y = generate_solar_data("https://www.cntk.ai/jup/dat/solar.csv",
                           TIMESTEPS, normalize=NORMALIZE)