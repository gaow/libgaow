#!/usr/bin/env python3
__author__ = "Gao Wang"
__copyright__ = "Copyright 2016, Stephens lab"
__email__ = "gaow@uchicago.edu"
__license__ = "MIT"
__version__ = "0.1.0"

import numpy as np, scipy as sp
from scipy.stats import norm, multivariate_normal as mvnorm

def inv_sympd(m):
    '''
    Inverse of symmetric positive definite
    https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi
    '''
    zz , _ = sp.linalg.lapack.dpotrf(m, False, False)
    inv_m , info = sp.linalg.lapack.dpotri(zz)
    # lapack only returns the upper or lower triangular part
    return np.triu(inv_m) + np.triu(inv_m, k=1).T

def get_svs(s, V):
    '''
    diag(s) @ V @ diag(s)
    '''
    return (s * V.T).T * s

class LikelihoodMASH:
    def __init__(self, data):
        self.J = data.B.shape[1]
        self.R = data.B.shape[0]
        self.P = len(data.U)
        self.data = data
        self.data.lik = {'relative_likelihood' : None,
                         'lfactor': None,
                         'marginal_loglik': None,
                         'loglik': None,
                         'null_loglik': None,
                         'alt_loglik': None}

    def compute_log10bf(self):
        self.data.log10bf = (self.data.lik['alt_loglik'] -  self.data.lik['null_loglik']) / np.log(10)

    def compute_relative_likelihood_matrix(self):
        matrix_llik = self._calc_likelihood_matrix_comcov() if self.data.is_common_cov() \
                      else self._calc_likelihood_matrix()
        lfactors = np.amax(matrix_llik, axis = 1)
        self.data.lik['relative_likelihood'] = np.exp(matrix_llik - lfactors)
        self.data.lik['lfactor'] = lfactors

    def _calc_likelihood_matrix(self):
        loglik = np.zeros((self.J, self.R))
        for j in range(self.J):
            sigma_mat = get_svs(self.data.S[:,j], self.data.V)
            loglik[j,:] = np.array([mvnorm.logpdf(self.data.B[:,j], cov = sigma_mat + self.data.U[p]) for p in range(self.P)])
        return loglik

    def _calc_likelihood_matrix_comcov(self):
        sigma_mat = get_svs(self.data.S[:,0], self.data.V)
        return np.matrix([mvnorm.logpdf(self.data.B, cov = sigma_mat + self.data.U[p]) for p in range(self.P)])

    def compute_loglik_from_matrix(self, options = ['all', 'alt', 'null']):
        '''
        data.lik.relative_likelihood first column is null, the rest are alt
        '''
        if 'marginal' in options:
            self.data.lik['marginal_loglik'] = np.log(self.data.lik['relative_likelihood'] @ self.data.pi) + self.data.lik['lfactor'] - np.sum(np.log(self.data.S), axis = 0)
            self.data.lik['loglik'] = np.sum(self.data.lik['marginal_loglik'])
        if 'alt' in options:
            self.data.lik['alt_loglik'] = np.log(self.data.lik['relative_likelihood'][:,1:] @ self.data.pi[1:] / (1 - self.data.pi[0])) + self.data.lik['lfactor'] - np.sum(np.log(self.data.S), axis = 0)
        if 'null' in options:
            self.data.lik['null_loglik'] = np.log(self.data.lik['relative_likelihood'][:,0]) + self.data.lik['lfactor'] - np.sum(np.log(self.data.S), axis = 0)


class PosteriorMASH:
    def __init__(self, data):
        '''
        // @param b_mat R by J
        // @param s_mat R by J
        // @param v_mat R by R
        // @param U_cube list of prior covariance matrices, for each mixture component P by R by R
        '''
        self.J = data.B.shape[1]
        self.R = data.B.shape[0]
        self.P = len(data.U)
        self.data = data
		self.data.post_mean_mat = np.matlib.zeros((self.R, self.J))
		self.data.post_mean2_mat = np.matlib.zeros((self.R, self.J))
        self.data.neg_prob_mat = np.matlib.zeros((self.R, self.J))
        self.data.zero_prob_mat = np.matlib.zeros((self.R, self.J))

    def compute_posterior_weights(self):
        d = (self.pi * self.data.lik['relative_likelihood'].T).T
        self.posterior_weights = d / np.sum(d, axis = 1)

    def compute_posterior(self):
        for j in range(self.J):
            Vinv_mat = inv_sympd(get_svs(self.data.S[:,j], self.data.V))
            mu1_mat = np.matlib.zeros((self.R, self.P))
            mu2_mat = np.matlib.zeros((self.R, self.P))
            zero_mat = np.matlib.zeros((self.R, self.P))
            neg_mat = np.matlib.zeros((self.R, self.P))
            for p in range(self.P):
                U1_mat = self.get_posterior_cov(Vinv_mat, self.data.U[p])
                mu1_mat[:,p] = self.get_posterior_mean(self.B[:,j], Vinv_mat, U1_mat)
                sigma_vec = np.sqrt(np.diag(U1_mat))
                mu2_mat[:,p] = np.square(mu1_mat[:,p]) + np.diag(U1_mat)
                neg_mat[:,p] = norm.cdf(mu1_mat[:,p], cov=sigma_vec)
                zero_mat[sigma_vec == 0,p] = 1.0
                neg_mat[sigma_vec == 0,p] = 0.0
            self.data.post_mean_mat[:,j] = mu1_mat * self.data.posterior_weights[:,j]
            self.data.post_mean2_mat[:,j] = mu2_mat * self.data.posterior_weights[:,j]
            self.data.neg_prob_mat[:,j] = neg_mat * self.data.posterior_weights[:,j]
            self.data.zero_prob_mat[:,j] = zero_mat * self.data.posterior_weights[:,j]

    def compute_posterior_comcov(self):
        Vinv_mat = inv_sympd(get_svs(self.data.S[:,0], self.data.V))
        mean_mat = np.matlib.zeros((self.R, self.J))
        for p in range(self.P):
            zero_mat = np.matlib.zeros((self.R, self.P))
            U1_mat = self.get_posterior_cov(Vinv_mat, self.data.U[p])
            mu1_mat = self.get_posterior_mean(self.B, Vinv_mat, U1_mat)
            sigma_vec = np.sqrt(np.diag(U1_mat))
            sigma_mat = np.repeat(sigma_vec, self.J, axis = 1)
            m2_mat = np.square(mu1_mat) + np.diag(U1_mat)
            neg_mat = norm.cdf(mu1_mat, mean_mat, sigma_mat)
            zero_mat[sigma_vec == 0,:] = 1.0
            neg_mat[sigma_vec == 0,:] = 0.0
            self.data.post_mean_mat += posterior_weights[p,:] * mu1_mat
            self.data.post_mean2_mat += posterior_weights[p,:] * mu2_mat
            self.data.neg_prob_mat += posterior_weights[p,:] * neg_mat
            self.data.zero_prob_mat += posterior_weights[p,:] * zero_mat

    @staticmethod
    def get_posterior_mean(B, V_inv, U):
        return U @ V_inv @ B

    @staticmethod
    def get_posterior_cov(V_inv, U):
        return U @ inv_sympd(V_inv @ U + np.identity(U.shape[0]))

    @classmethod
    def apply(cls, data):
        obj = cls(data)
        obj.compute_posterior_weights()
        if data.is_common_cov():
            obj.compute_posterior_comcov()
        else:
            obj.compute_posterior()

class PriorMASH:
    def __init__(self, data):
        self.data = data
        self.R = data.B.shape[0]

    def expand_cov(self, grid, use_pointmass = True):
        self.data.U = dict(sum([[(f"{p}.{i}", g) for i, g in enumerate(self.data.U[p] * np.square(grid))] for p in self.data.U], []))
        if use_pointmass:
            self.data.U['null'] = np.matlib.zeros((self.R, self.R))
