#!/usr/bin/env python3
__author__ = "Gao Wang"
__copyright__ = "Copyright 2016, Stephens lab"
__email__ = "gaow@uchicago.edu"
__license__ = "MIT"
__version__ = "0.1.0"

from .model_mash import PriorMASH, LikelihoodMASH, PosteriorMASH
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
        self.x_centered = self.y_centered = self.z_centered = False
        self.reset({'X': X, 'Y': Y, 'Z': Z, 'B': B, 'S': S, 'lik' : None})
        if X is not None:
            self.trace_XXt = np.sum(np.square(X), axis = 1)
        if (self.X is not None and self.Y is not None) and (self.B is None and self.S is None):
            self.get_summary_stats()

    def fit(self):
        pass

    def get_summary_stats(self):
        if self.Z is not None:
            self.remove_covariates()
        # Compute betahat
        XtY = self.X.T @ self.Y
        XtX_vec = np.einsum('ji,ji->i', self.X, self.X)
        self.B = XtY / XtX_vec[:,np.newaxis]
        # Compute se(betahat)
        Xr = self.Y - np.einsum('ij,jk->jik', self.X, self.B)
        Re = np.einsum('ijk,ijk->ik', Xr, Xr)
        self.S = np.sqrt(Re / XtX_vec[:,np.newaxis] / (self.X.shape[0] - 2))

    def remove_covariates(self):
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

    def set_prior(self, U, grid = None, pi = None):
        # FIXME: allow autogrid select?
        # FIXME: ensure U is ordered dict
        self.U = U
        self.grid = grid
        self.pi = np.array(pi) if pi is not None else None
        prior = PriorMASH(self)
        if grid is not None:
            prior.expand_cov()

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

    def set_prior(self, U, grid = None, pi = None):
        self.mash.set_prior(U, grid, pi)

    def fit(self, niter=100, L=5, calc_elbo=False):
        self.alpha0 = np.zeros((L, self.mash.X.shape[1]))
        self.mu0 = np.zeros((L, self.mash.X.shape[1], self.Y.shape[1]))
        for i in range(niter):
            self._calc_update()
            if calc_elbo:
                self._calc_elbo()
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
        bf = np.exp(self.mash.l10bf)
        return bf/np.sum(bf), self.mash.post_mean_mat.T

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
    model.fit(niter = 50)
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.scatter([x+1 for x in range(len(model.post_mean_mat[:,0]))], model.post_mean_mat[:,0],
                cmap="viridis")
    ax = plt.gca()
    plt.show()
    plt.scatter([x+1 for x in range(len(model.post_mean_mat[:,1]))], model.post_mean_mat[:,1],
                cmap="viridis")
    ax = plt.gca()
    plt.show()
