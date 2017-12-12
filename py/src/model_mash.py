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

def safe_mvnorm_logpdf(val, cov):
    try:
        return mvnorm.logpdf(val, cov=cov)
    except np.linalg.linalg.LinAlgError:
        if len(val.shape) == 1:
            return np.inf if np.sum(val) < 1E-6 else -np.inf
        else:
            return np.array([np.inf if np.sum(x) < 1E-6 else -np.inf for x in val.T])

class LikelihoodMASH:
    def __init__(self, data):
        self.J = data.B.shape[0]
        self.R = data.B.shape[1]
        self.P = len(data.U)
        self.data = data

    def compute_log10bf(self):
        self.data.l10bf = (self.data.lik['alt_loglik'] -  self.data.lik['null_loglik']) / np.log(10)

    def compute_relative_likelihood_matrix(self):
        matrix_llik = self._calc_likelihood_matrix_comcov() if self.data.is_common_cov() \
                      else self._calc_likelihood_matrix()
        lfactors = np.amax(matrix_llik, axis = 1)
        self.data.lik['relative_likelihood'] = np.exp(matrix_llik - np.vstack(lfactors))
        self.data.lik['lfactor'] = lfactors

    def _calc_likelihood_matrix(self):
        loglik = np.zeros((self.J, self.P))
        for j in range(self.J):
            sigma_mat = get_svs(self.data.S[j,:], self.data.V)
            loglik[j,:] = np.array([safe_mvnorm_logpdf(self.data.B[j,:], sigma_mat + self.data.U[p]) for p in self.data.U])
        return loglik

    def _calc_likelihood_matrix_comcov(self):
        sigma_mat = get_svs(self.data.S[0,:], self.data.V)
        return np.array([safe_mvnorm_logpdf(self.data.B, sigma_mat + self.data.U[p]) for p in self.data.U]).T

    def compute_loglik_from_matrix(self, options = ['marginal', 'alt', 'null']):
        '''
        data.lik.relative_likelihood first column is null, the rest are alt
        '''
        if 'marginal' in options:
            # add a very small number for numeric issue eg prevent log(zero)
            # delta = 1 / np.finfo(float).max
            delta = 0
            self.data.lik['marginal_loglik'] = np.log(self.data.lik['relative_likelihood'] @ self.data.pi + delta) + self.data.lik['lfactor']
            self.data.lik['loglik'] = np.sum(self.data.lik['marginal_loglik'])
        if 'alt' in options:
            self.data.lik['alt_loglik'] = np.log(self.data.lik['relative_likelihood'][:,1:] @ (self.data.pi[1:] / (1 - self.data.pi[0])) + delta) + self.data.lik['lfactor']
        if 'null' in options:
            self.data.lik['null_loglik'] = np.log(self.data.lik['relative_likelihood'][:,0] + delta) + self.data.lik['lfactor']

class PosteriorMASH:
    def __init__(self, data):
        '''
        // @param b_mat J by R
        // @param s_mat J by R
        // @param v_mat R by R
        // @param U_cube list of prior covariance matrices, for each mixture component P by R by R
        '''
        self.J = data.B.shape[0]
        self.R = data.B.shape[1]
        self.P = len(data.U)
        self.data = data
        self.data.post_mean_mat = np.zeros((self.R, self.J))
        self.data.post_mean2_mat = np.zeros((self.R, self.J))
        self.data.neg_prob_mat = np.zeros((self.R, self.J))
        self.data.zero_prob_mat = np.zeros((self.R, self.J))

    def compute_posterior_weights(self):
        d = (self.data.pi * self.data.lik['relative_likelihood'])
        self.data.posterior_weights = (d.T / np.sum(d, axis = 1))

    def compute_posterior(self):
        for j in range(self.J):
            Vinv_mat = inv_sympd(get_svs(self.data.S[j,:], self.data.V))
            mu1_mat = np.zeros((self.R, self.P))
            mu2_mat = np.zeros((self.R, self.P))
            zero_mat = np.zeros((self.R, self.P))
            neg_mat = np.zeros((self.R, self.P))
            for p, name in enumerate(self.data.U.keys()):
                U1_mat = self.get_posterior_cov(Vinv_mat, self.data.U[name])
                mu1_mat[:,p] = self.get_posterior_mean_vec(self.data.B[j,:], Vinv_mat, U1_mat)
                sigma_vec = np.sqrt(np.diag(U1_mat))
                null_cond = (sigma_vec == 0)
                mu2_mat[:,p] = np.square(mu1_mat[:,p]) + np.diag(U1_mat)
                if not null_cond.all():
                    neg_mat[np.invert(null_cond),p] = norm.sf(mu1_mat[np.invert(null_cond),p], scale=sigma_vec[np.invert(null_cond)])
                zero_mat[null_cond,p] = 1.0
            self.data.post_mean_mat[:,j] = mu1_mat @ self.data.posterior_weights[:,j]
            self.data.post_mean2_mat[:,j] = mu2_mat @ self.data.posterior_weights[:,j]
            self.data.neg_prob_mat[:,j] = neg_mat @ self.data.posterior_weights[:,j]
            self.data.zero_prob_mat[:,j] = zero_mat @ self.data.posterior_weights[:,j]

    def compute_posterior_comcov(self):
        Vinv_mat = inv_sympd(get_svs(self.data.S[0,:], self.data.V))
        for p, name in enumerate(self.data.U.keys()):
            zero_mat = np.zeros((self.R, self.J))
            U1_mat = self.get_posterior_cov(Vinv_mat, self.data.U[name])
            mu1_mat = self.get_posterior_mean_mat(self.data.B, Vinv_mat, U1_mat)
            sigma_vec = np.sqrt(np.diag(U1_mat))
            null_cond = (sigma_vec == 0)
            sigma_mat = np.tile(sigma_vec, (self.J, 1)).T
            neg_mat = np.zeros((self.R, self.J))
            if not null_cond.all():
                neg_mat[np.invert(null_cond),:] = norm.sf(mu1_mat[np.invert(null_cond),:], scale = sigma_mat[np.invert(null_cond),:])
            mu2_mat = np.square(mu1_mat) + np.vstack(np.diag(U1_mat))
            zero_mat[null_cond,:] = 1.0
            self.data.post_mean_mat += self.data.posterior_weights[p,:] * mu1_mat
            self.data.post_mean2_mat += self.data.posterior_weights[p,:] * mu2_mat
            self.data.neg_prob_mat += self.data.posterior_weights[p,:] * neg_mat
            self.data.zero_prob_mat += self.data.posterior_weights[p,:] * zero_mat

    @staticmethod
    def get_posterior_mean_vec(B, V_inv, U):
        return U @ (V_inv @ B)

    @staticmethod
    def get_posterior_mean_mat(B, V_inv, U):
        return (B @ V_inv @ U).T

    @staticmethod
    def get_posterior_cov(V_inv, U):
        return U @ np.linalg.inv(V_inv @ U + np.identity(U.shape[0]))

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
        self.R = data.U[list(data.U.keys())[0]].shape[1]

    def expand_cov(self, use_pointmass = True):
        def product(x,y):
            for item in y:
                yield x*item
        # dict in Python 3.6 is ordered
        res = dict()
        if use_pointmass:
            res['null'] = np.zeros((self.R, self.R))
        res.update(dict(sum([[(f"{p}.{i+1}", g) for i, g in enumerate(product(self.data.U[p], np.square(self.data.grid)))] for p in self.data.U], [])))
        self.data.U = res
