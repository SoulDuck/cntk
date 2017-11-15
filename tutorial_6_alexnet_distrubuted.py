import cntk as C

import os
import math
import argparse
import cntk as C
import _cntk_py

from cntk.logging import *
from cntk.train.training_session import *
from cntk import *
from cntk.train.distributed import *
from cntk.io import ImageDeserializer ,MinibatchSource , StreamDef ,StreamDefs , FULL_DATA_SWEEP
import cntk.io.transforms as xForms
from cntk.layers import Convolution2D, Activation, MaxPooling, Dense, Dropout, default_options, Sequential
from cntk.initializer import normal


# default Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(abs_path, "Models")
log_dir = None

# model dimensions
image_height = 227
image_width  = 227
num_channels = 3  # RGB
num_classes  = 1000
model_name   = "AlexNet.model"


def create_image_mb_source(map_file, is_training, total_number_of_samples):
    if not os.path.exists(map_file):
        raise RuntimeError("File '%s' does not exist." %map_file)

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if is_training:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.88671875, jitter_type='uniratio') # train uses jitter
        ]
    else:
        transforms += [
            xforms.crop(crop_type='center', side_ratio=0.88671875) # test has no jitter
        ]

    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
    ]

    # deserializer
    return MinibatchSource(
        ImageDeserializer(map_file, StreamDefs(
            features=StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
            labels=StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize=is_training,
        max_samples=total_number_of_samples,
        multithreaded_deserializer=True)




def LocalResponseNormalization(k, n, alpha, beta, name=''):
    x = C.placeholder(name='lrn_arg')
    x2 = C.square(x)
    # reshape to insert a fake singleton reduction dimension after the 3th axis (channel axis). Note Python axis order and BrainScript are reversed.
    x2s = C.reshape(x2, (1, C.InferredDimension), 0, 1)
    W = C.constant(alpha/(2*n+1), (1,2*n+1,1,1), name='W')
    # 3D convolution with a filter that has a non 1-size only in the 3rd axis, and does not reduce since the reduction dimension is fake and 1
    y = C.convolution (W, x2s)
    # reshape back to remove the fake singleton reduction dimension
    b = C.reshape(y, C.InferredDimension, 0, 2)
    den = C.exp(beta * C.log(k + b))
    apply_x = C.element_divide(x, den)
    return apply_x

