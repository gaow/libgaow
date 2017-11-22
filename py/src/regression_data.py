#!/usr/bin/env python3
__author__ = "Gao Wang"
__copyright__ = "Copyright 2016, Stephens lab"
__email__ = "gaow@uchicago.edu"
__license__ = "MIT"
__version__ = "0.1.0"

class RegressionData:
    def __init__(self, X, Y, Z = None):
        self.X = X
        self.Y = Y
        self.Z = Z
