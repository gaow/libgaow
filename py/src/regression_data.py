#!/usr/bin/env python3
__author__ = "Gao Wang"
__copyright__ = "Copyright 2016, Stephens lab"
__email__ = "gaow@uchicago.edu"
__license__ = "MIT"
__version__ = "0.1.0"

from .model_mash import PriorMASH, LikelihoodMASH, PosteriorMASH
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import numpy as np
import copy

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo):
        return dotdict(copy.deepcopy(dict(self)))

class RegressionData(dotdict):
    def __init__(self, X = None, Y = None, Z = None, B = None, S = None):
        # FIXME: check if inputs are indeed numpy arrays
        self.reset({'X': X, 'Y': Y, 'Z': Z, 'B': B, 'S': S, 'lik' : None})
        if X is not None:
            self.trace_XXt = np.sum(np.square(X), axis = 1)
        if (self.X is not None and self.Y is not None) and (self.B is None and self.S is None):
            self.get_summary_stats()

    def fit(self):
        pass

    def get_summary_stats(self):
        '''
        perform univariate regression
        FIXME: it is slower than lapply + .lm.fit in R
        FIXME: this faster implementation is on my watch list:
        https://github.com/ajferraro/fastreg
        '''
        self.B = np.zeros((self.X.shape[1], self.Y.shape[1]))
        self.S = np.zeros((self.X.shape[1], self.Y.shape[1]))
        for r, y in enumerate(self.Y.T):
            self.B[:,r], self.S[:,r] = self.univariate_simple_regression(self.X, y)[:,[0,2]].T

    def reset(self, init_data):
        self.update(init_data)

    @staticmethod
    def univariate_simple_regression(X, y, Z=None):
        if Z is not None:
            model = LinearRegression()
            model.fit(Z, y)
            y = y - model.predict(Z)
        return np.vstack([linregress(x, y) for x in X.T])[:,[0,1,4]]

    def __str__(self):
        l = dir(self)
        d = self.__dict__
        from pprint import pformat
        return pformat(d, indent = 4)

class MASH(RegressionData):
    def __init__(self, X = None, Y = None, Z = None, B = None, S = None, V = None):
        RegressionData.__init__(self, X, Y, Z, B, S)
        self.post_mean_mat = None
        self.post_mean2_mat = None
        self.neg_prob_mat = None
        self.zero_prob_mat = None
        self._is_common_cov = None
        self.V = V
        self.U = None
        self.pi = None
        self.posterior_weights = None
        self.grid = None
        self.l10bf = None

    def fit(self):
        if self.pi is None:
            raise RuntimeError('MASH mixture fitting not implemented')
        if self.V is None:
            self.V = np.cov(self.Y, rowvar = False)
        lik = LikelihoodMASH(self)
        lik.compute_relative_likelihood_matrix()
        lik.compute_loglik_from_matrix()
        lik.compute_log10bf()
        PosteriorMASH.apply(self)

    def is_common_cov(self):
        if self._is_common_cov is None and self.S is not None:
            self._is_common_cov = (self.S.T == self.S.T[0,:]).all()
        return self._is_common_cov

    def set_prior(self, U, grid, pi = None):
        # FIXME: allow autogrid select?
        self.U = U
        self.grid = grid
        self.pi = np.array(pi) if pi is not None else None
        prior = PriorMASH(self)
        prior.expand_cov()
