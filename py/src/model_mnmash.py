#!/usr/bin/env python3
__author__ = "Gao Wang"
__copyright__ = "Copyright 2016, Stephens lab"
__email__ = "gaow@uchicago.edu"
__license__ = "MIT"
__version__ = "0.1.0"

import numpy as np, scipy as sp
from collections import OrderedDict
from scipy.stats import norm, multivariate_normal as mvnorm

class ELBOMNM:
    def __init__(self, data):
        self.data = data
