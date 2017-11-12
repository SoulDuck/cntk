# -*- coding:utf-8 -*-
import input_stock_data
import numpy as np
import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

#Lets build the network
num_days_back = 8 # 이게 input py에서 가져온거다. 8일 전까지 확인 하기 때문에...
input_dim = 2 + num_days_back
num_output_classes = 2 #Remember we need to have 2 since we are trying to classify if the market goes up or down 1 hot encoded
num_hidden_layers = 2
hidden_layers_dim = 2 + num_days_back
input_dynamic_axes = [C.Axis.default_batch_axis()]

input = C.input_variable(input_dim, dynamic_axes=input_dynamic_axes)
label = C.input_variable(num_output_classes, dynamic_axes=input_dynamic_axes)

print input
print label

def create_model(input, num_output_classes):

    h = input
    with C.layers.default_options(init = C.glorot_uniform()):
        for i in range(num_hidden_layers):
            h = C.layers.Dense(hidden_layers_dim,
                               activation = C.relu)(h)
        r = C.layers.Dense(num_output_classes, activation=None)(h)
    return r

z = create_model(input, num_output_classes)
print z
loss = C.cross_entropy_with_softmax(z, label)
label_error = C.classification_error(z, label)
lr_per_minibatch = C.learning_rate_schedule(0.125,C.UnitType.minibatch)
trainer = C.Trainer(z, (loss, label_error), [C.sgd(z.parameters, lr=lr_per_minibatch)])
print 'a'
training_features, training_labels, training_data , test_data=input_stock_data.make_stock_input()
#Initialize the parameters for the trainer, we will train in large minibatches in sequential order
minibatch_size = 50 #100
num_minibatches = len(training_data.index) // minibatch_size

#Run the trainer on and perform model training
training_progress_output_freq = 1

# Visualize the loss over minibatch
plotdata = {"batchsize":[], "loss":[], "error":[]}



# Train our neural network
tf = np.split(training_features,num_minibatches)
tl = np.split(training_labels, num_minibatches)

for i in range(num_minibatches*num_passes): # multiply by the
    features = np.ascontiguousarray(tf[i%num_minibatches])
    labels = np.ascontiguousarray(tl[i%num_minibatches])

    # Specify the mapping of input variables in the model to actual minibatch data to be trained with
    trainer.train_minibatch({input : features, label : labels})
    batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)
    if not (loss == "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)