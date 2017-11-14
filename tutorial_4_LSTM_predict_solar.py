import math
import os
import matplotlib
if "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import input_solar
import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)


TIMESTEPS=14
# process batches of 10 days
BATCH_SIZE = TIMESTEPS * 10

NORMALIZE = 20000
X, Y = input_solar.generate_solar_data("https://www.cntk.ai/jup/dat/solar.csv",
                           TIMESTEPS, normalize=NORMALIZE)
def next_batch(x, y, ds):
    """get the next batch for training"""

    def as_batch(data, start, count):
        return data[start:start + count]

    for i in range(0, len(x[ds]), BATCH_SIZE):
        yield as_batch(X[ds], i, BATCH_SIZE), as_batch(Y[ds], i, BATCH_SIZE)



#Specify the internal-state dimensions of the LSTM cell
H_DIMS = 15
def create_model(x):
    """Create the model for time series prediction"""
    with C.layers.default_options(initial_state = 0.1):
        m = C.layers.Recurrence(C.layers.LSTM(H_DIMS))(x)
        m = C.sequence.last(m)
        m = C.layers.Dropout(0.2)(m)
        m = C.layers.Dense(1)(m)
        return m


def training(EPOCHS=100):
    # input sequences
    x = C.sequence.input_variable(1)
    print 'x shape : {}'.format(str(x))

    # create the model
    z = create_model(x)

    # expected output (label), also the dynamic axes of the model output
    # is specified as the model of the label input
    l = C.input_variable(1, dynamic_axes=z.dynamic_axes, name="y")

    # the learning rate
    learning_rate = 0.005
    lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)

    # loss function
    loss = C.squared_error(z, l)

    # use squared error to determine error for now
    error = C.squared_error(z, l)

    # use adam optimizer
    momentum_time_constant = C.momentum_as_time_constant_schedule(BATCH_SIZE / -math.log(0.9))
    learner = C.fsadagrad(z.parameters,
                          lr = lr_schedule,
                          momentum = momentum_time_constant)
    trainer = C.Trainer(z, (loss, error), [learner])



    loss_summary = []
    #Get Error
    start = time.time()
    for epoch in range(0, EPOCHS):
        for x_batch, l_batch in next_batch(X, Y, "train"):
            trainer.train_minibatch({x: x_batch, l: l_batch})

        if epoch % (EPOCHS / 10) == 0:
            training_loss = trainer.previous_minibatch_loss_average
            loss_summary.append(training_loss)
            print("epoch: {}, loss: {:.4f}".format(epoch, training_loss))
    # Visualization
    for j, ds in enumerate(["val", "test"]):
        results = []
        for x_batch, _ in next_batch(X, Y, ds):
            pred = z.eval({x: x_batch})
            results.extend(pred[:, 0])
        # because we normalized the input data we need to multiply the prediction
        # with SCALER to get the real values.
        a[j].plot((Y[ds] * NORMALIZE).flatten(), label=ds + ' raw');
        a[j].plot(np.array(results) * NORMALIZE, label=ds + ' pred');
        a[j].legend();
    plt.savefig('./plot_solar_pred.png')
    print("Training took {:.1f} sec".format(time.time() - start))
# validate
def get_mse(X,Y,labeltxt):
    result = 0.0
    for x1, y1 in next_batch(X, Y, labeltxt):
        eval_error = trainer.test_minibatch({x : x1, l : y1})
        result += eval_error
    return result/len(X[labeltxt])
# predict
f, a = plt.subplots(2, 1, figsize=(12, 8))
if __name__ == '__main__':
    training()