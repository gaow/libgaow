#!/usr/bin/env python3
__author__ = "Gao Wang"
__copyright__ = "Copyright 2016, Stephens lab"
__email__ = "gaow@uchicago.edu"
__license__ = "MIT"
__version__ = "0.1.0"

from regression_lik import *
from regression_prior import *
from regression_post import *

class RegressionData:
    def __init__(self, X = None, Y = None, Z = None):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.B = None
        self.S = None
        self.prior = None
        self.loglik = None
        self.BF = None
        self.posterior = None

    def set_prior(self):
        pass

    def calc_loglik(self):
        pass

    def calc_bf(self):
        pass

class MashData(RegressionData):
    def __init__(self, X = None, Y = None, Z = None):
        RegressionData.__init__(self, X, Y, Z)
