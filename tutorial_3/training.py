# -*- coding:utf-8 -*-
import input
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

exit()


def create_model(input, num_output_classes):

    h = input
    with C.layers.default_options(init = C.glorot_uniform()):
        for i in range(num_hidden_layers):
            h = C.layers.Dense(hidden_layers_dim,
                               activation = C.relu)(h)
        r = C.layers.Dense(num_output_classes, activation=None)(h)
    return r

z = create_model(input, num_output_classes)
loss = C.cross_entropy_with_softmax(z, label)
label_error = C.classification_error(z, label)
lr_per_minibatch = C.learning_rate_schedule(0.125,C.UnitType.minibatch)
trainer = C.Trainer(z, (loss, label_error), [C.sgd(z.parameters, lr=lr_per_minibatch)])