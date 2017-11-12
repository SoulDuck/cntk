import input_stock_data
import numpy as np

training_features, training_labels, training_data , test_data=input_stock_data.make_stock_input()
print np.shape(training_features[:-1])
print np.split(training_features , 10)