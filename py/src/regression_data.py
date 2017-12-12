#!/usr/bin/env python3
__author__ = "Gao Wang"
__copyright__ = "Copyright 2016, Stephens lab"
__email__ = "gaow@uchicago.edu"
__license__ = "MIT"
__version__ = "0.1.0"

from .model_mash import PriorMASH, LikelihoodMASH, PosteriorMASH
import numpy as np
import os, copy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __deepcopy__(self, memo):
        return dotdict(copy.deepcopy(dict(self)))

class RegressionData(dotdict):
    def __init__(self, X = None, Y = None, Z = None, B = None, S = None):
        # FIXME: check if inputs are indeed numpy arrays
        self.debug = dotdict()
        self.x_centered = self.y_centered = self.z_centered = False
        self.reset({'X': X, 'Y': Y, 'Z': Z, 'B': B, 'S': S, 'lik' : None})
        if X is not None:
            self.trace_XXt = np.sum(np.square(X), axis = 1)
        if (self.X is not None and self.Y is not None) and (self.B is None and self.S is None):
            self.get_summary_stats()
        self.xcorr = None
        self.sigma = None

    def fit(self):
        pass

    def get_summary_stats(self):
        if self.Z is not None:
            self.remove_covariates()
        # Compute betahat
        XtX_vec = np.einsum('ji,ji->i', self.X, self.X)
        self.B = (self.X.T @ self.Y) / XtX_vec[:,np.newaxis]
        # Compute se(betahat)
        Xr = self.Y - np.einsum('ij,jk->jik', self.X, self.B)
        Re = np.einsum('ijk,ijk->ik', Xr, Xr)
        self.S = np.sqrt(Re / XtX_vec[:,np.newaxis] / (self.X.shape[0] - 2))

    def remove_covariates(self):
        if self.Z is not None:
            self.Y -= self.Z @ (np.linalg.inv(self.Z.T @ self.Z) @ self.Z.T @ self.Y)
            self.Z = None

    def reset(self, init_data):
        self.update(init_data)
        if 'X' in init_data:
            self.x_centered = False
        if 'Y' in init_data:
            self.y_centered = False
        if 'Z' in init_data:
            self.z_centered = False
        if self.X is not None and not self.x_centered:
            self.X -= np.mean(self.X, axis=0, keepdims=True)
            self.x_centered = True
        if self.Y is not None and not self.y_centered:
            self.Y -= np.mean(self.Y, axis=0, keepdims=True)
            self.y_centered = True
        if self.Z is not None and not self.z_centered:
            self.Z -= np.mean(self.Z, axis=0, keepdims=True)
            self.z_centered = True

    def get_xcorr(self, save_to = None):
        '''
        compute column correlations of X.
        i.e., LD if X is genotype matrix based on r^2
        - the result is signed r^2 values of Pearson correlations
        '''
        self.xcorr = np.corrcoef(self.X, rowvar = False)
        self.xcorr = (np.square(self.xcorr) * np.sign(self.xcorr)).astype(np.float16)
        if save_to is not None:
            if os.path.isfile(save_to):
                os.remove(save_to)
            np.save(save_to, self.xcorr)

    def set_xcorr(self, xcorr):
        self.xcorr = xcorr

    def plot_xcorr(self, out):
        use_abs = np.sum((self.xcorr < 0).values.ravel()) == 0
        fig, ax = plt.subplots()
        cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=1, dark=0, as_cmap=True)
        sns.heatmap(self.xcorr, ax = ax, cmap = cmap, vmin=-1 if not use_abs else 0,
                    vmax=1, square=True, xticklabels = False, yticklabels = False)
        ax = plt.gca()
        plt.savefig(out, dpi = 500)

    def permute_X(self):
        '''
        Permute X columns, ie break LD structure for genotype input X
        '''
        np.random.shuffle(self.X)

    def plot_B(self, b_vec, out):
        fig, ax = plt.subplots()
        sns.lmplot('index', 'data', data = pd.DataFrame({'index': [x+1 for x in range(len(b_vec))],
                                         'data': b_vec}))
        ax = plt.gca()
        plt.savefig(out, dpi = 500)

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
        self.lik = {'relative_likelihood' : None,
                         'lfactor': None,
                         'marginal_loglik': None,
                         'loglik': None,
                         'null_loglik': None,
                         'alt_loglik': None}

    def fit(self):
        if self.pi is None:
            raise RuntimeError('MASH mixture fitting not implemented')
        if self.V is None:
            self.V = np.corrcoef(self.Y, rowvar = False)
        lik = LikelihoodMASH(self)
        lik.compute_relative_likelihood_matrix()
        lik.compute_loglik_from_matrix()
        lik.compute_log10bf()
        PosteriorMASH.apply(self)

    def is_common_cov(self):
        if self._is_common_cov is None and self.S is not None:
            self._is_common_cov = (self.S.T == self.S.T[0,:]).all()
        return self._is_common_cov

    def set_prior(self, U, grid = None, pi = None, use_pointmass = True):
        # FIXME: allow autogrid select?
        # FIXME: ensure U is ordered dict
        self.U = U
        self.grid = grid
        self.pi = np.array(pi) if pi is not None else None
        prior = PriorMASH(self)
        if grid is not None:
            prior.expand_cov(use_pointmass)

class MNMASH:
    def __init__(self, X=None, Y=None, Z=None, B=None, S=None, V=None):
        self.mash = MASH(X=X,Y=Y,Z=Z,B=B,S=S,V=np.corrcoef(Y, rowvar=False) if V is None else V)
        self.Y = Y
        # variational parameter for the one-nonzero effect model for each l and j
        self.alpha0 = None
        # posterior mean on \beta_lj
        self.mu0 = None
        self.Xr0 = np.zeros((self.Y.shape[0], self.Y.shape[1]))
        self.elbo = []
        self.post_mean_mat = None
        self.iter_id = 0

    def set_prior(self, U, grid = None, pi = None, use_pointmass = True):
        self.mash.set_prior(U, grid, pi, use_pointmass)

    def fit(self, niter=50, L=5, bool_elbo=False):
        self.alpha0 = np.zeros((L, self.mash.X.shape[1]))
        self.mu0 = np.zeros((L, self.mash.X.shape[1], self.Y.shape[1]))
        for i in range(niter):
            self._calc_update()
            if bool_elbo:
                self._calc_elbo()
            self.iter_id += 1
        self._calc_posterior()

    def _calc_update(self):
        for l in range(self.alpha0.shape[0]):
            self.Xr0 -= self.mash.X @ (np.vstack(self.alpha0[l,:]) * self.mu0[l,:,:])
            self.alpha0[l,:], self.mu0[l,:,:] = self._calc_single_snp(self.Y - self.Xr0)
            self.Xr0 += self.mash.X @ (np.vstack(self.alpha0[l,:]) * self.mu0[l,:,:])

    def _calc_single_snp(self, R):
        self.mash.reset({'Y': R})
        self.mash.get_summary_stats()
        self.mash.fit()
        bf_rel = np.exp((self.mash.l10bf - np.max(self.mash.l10bf)) * np.log(10))
        return bf_rel / np.sum(bf_rel), self.mash.post_mean_mat.T

    def _calc_elbo(self):
        pass

    def _calc_posterior(self):
        almu = np.zeros((self.mu0.shape[0], self.mu0.shape[1], self.mu0.shape[2]))
        for l in range(self.alpha0.shape[0]):
            almu[l,:,:] = np.vstack(self.alpha0[l,:]) * self.mu0[l,:,:]
        self.post_mean_mat = np.sum(almu, axis = 0)

    def __str__(self):
        l = dir(self)
        d = self.__dict__
        from pprint import pformat
        return pformat(d, indent = 4)

if __name__ == '__main__':
    model = MNMASH(X=X,Y=Y)
    model.fit(niter = 10)
