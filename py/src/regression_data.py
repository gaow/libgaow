#!/usr/bin/env python3
__author__ = "Gao Wang"
__copyright__ = "Copyright 2016, Stephens lab"
__email__ = "gaow@uchicago.edu"
__license__ = "MIT"
__version__ = "0.1.0"

from .model_mash import PriorMASH, LikelihoodMASH, PosteriorMASH
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

class RegressionData:
    def __init__(self, X = None, Y = None, Z = None, B = None, S = None):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.B = B
        self.S = S
        self.lik = None
        self.l10bf = None
        if (self.X is not None and self.Y is not None) and (self.B is None and self.S is None):
            self.get_summary_stats()

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
            self.B[:,r], self.S[:,r] = univariate_simple_regression(self.X, y)[:,[0,2]].T
 
    def set_prior(self):
        pass

    def calc_likelihood(self):
        pass

    def calc_posterior(self):
        pass

    def calc_bf(self):
        pass

    @staticmethod
    def univariate_simple_regression(X, y, Z=None):
        if Z is not None:
            model = LinearRegression()
            model.fit(Z, y)
            y = y - model.predict(Z)
        return np.vstack([linregress(x, y) for x in X.T])[:,[0,1,4]]


class MASHData(RegressionData):
    def __init__(self, X = None, Y = None, Z = None, B = None, S = None):
        RegressionData.__init__(self, X, Y, Z, B, S)
        self.post_mean_mat = None
        self.post_mean2_mat = None
        self.neg_prob_mat = None
        self.zero_prob_mat = None
        self._is_common_cov = None
        self.V = None
        self.U = None
        self.pi = None
        self.posterior_weights = None
        self.grid = None

    def is_common_cov(self):
        if self._is_common_cov is None and self.S is not None:
            self._is_common_cov = (self.S.T == self.S.T[0,:]).all()
        return self._is_common_cov

    def calc_posterior(self):
        PosteriorMASH.apply(self)

    def calc_likelihood(self):
        LikelihoodMASH.apply(self)
