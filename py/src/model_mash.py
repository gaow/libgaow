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
        self.data.lik = np.matlib.zeros(self.J, self.P)
        self.data.loglik = np.matlib.zeros(self.J, self.P)

    def compute_likelihood(self):
        mean_vec = np.zeros(self.R)
        for j in range(self.J):
            sigma_mat = get_svs(self.data.S[:,j], self.data.V)
            for p in range(self.P):
                self.data.loglik[j,p] = mvnorm.logpdf(self.data.B[:,j], mean_vec, sigma_mat + self.data.U[p])
        self.data.lik = np.exp(self.data.loglik)

    def compute_likelihood_comcov(self):
        mean_vec = np.zeros(self.R)
        sigma_mat = get_svs(self.data.S[:,0], self.data.V)
        for p in range(self.P):
            self.data.loglik[:,p] = mvnorm.logpdf(self.data.B, mean_vec, sigma_mat + self.data.U[p])

    @classmethod
    def apply(cls, data):
        obj = cls(data)
        if data.is_common_cov():
            obj.compute_likelihood_comcov()
        else:
            obj.compute_likelihood()

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

    def compute_posterior(self):
        mean_vec = np.zeros(self.R)
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
                neg_mat[:,p] = norm.cdf(mu1_mat[:,p], mean_vec, sigma_vec)
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
        if data.is_common_cov():
            obj.compute_posterior_comcov()
        else:
            obj.compute_posterior()
