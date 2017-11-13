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

import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
