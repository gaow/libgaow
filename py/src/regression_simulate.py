#!/usr/bin/env python3
__author__ = "Gao Wang"
__copyright__ = "Copyright 2016, Stephens lab"
__email__ = "gaow@uchicago.edu"
__license__ = "MIT"
__version__ = "0.1.0"

from .regression_data import MASH

import numpy as np

class MMRegressionSimulator(MASH):
    def __init__(self, X):
        MASH.__init__(self, X=X, Y=None, Z=None, B=None, S=None, V=None)

    def generate_B(self, set_nonzero = None, number_nonzero = None):
        '''
        Generate B under B_{j\cdot} ~ \sum \pi_i N_R(0, U),
        with sparsity:
        - set_nonzero can be
          - a list of index: effects of SNPs within the list are marked non-zero.
            this help ensuring true effects occur in different LD blocks, creating
            a simpler case where causal variants are not convoluted
          - a probability: with this probability, an effect is marked non-zero
        - number_nonzero:
          - overwrites set_nonzero when set_nonzero is a probability
          - or takes effect after set_nonzero when a list is given, in which case number_nonzero effects are selected from the list
        '''
        self.B = np.zeros((self.X.shape[1], self.U[0].shape[0]))
        js = list(range(self.B.shape[0])) if not isinstance(set_nonzero, list) else set_nonzero
        if isinstance(number_nonzero, int):
            if len(js) < number_nonzero:
                number_nonzero = len(js)
            js = np.random.choice(js, number_nonzero, replace = False)
        elif isinstance(set_nonzero, float):
            js = [y for x, y in zip(np.random.binomial(1, set_nonzero, size = len(js)), js) if x > 0]
        else:
            pass
        mus = np.zeros(self.B.shape[1])
        for j in js:
            sigma = self.U[np.random.multinomial(1, self.pi, size = 1).tolist()[0].index(1)]
            self.B[j,:] = np.random.multivariate_normal(mus, sigma, 1)

    def generate_Y(self, sigma):
        self.sigma = sigma
        self.Y = self.X @ self.B + np.random.multivariate_normal(np.zeros(self.B.shape[1]), sigma)

    def select_independent_snps(self, cutoff1 = 0.8, cutoff2 = 10, cutoff3 = 0.02):
        '''
        Based on LD matrix select SNPs in strong LD with other SNPs
        yet are independent among this selected set.
        - cutoff1: definition of LD block -- LD have to be > cutoff1
        - cutoff2: define a large enough block -- block size have to be > cutoff2 / 0.8
        - cutoff3: now select LD that are completely independent
        '''
        assert self.xcorr is not None
        print('Count strong LD')
        import pandas as pd
        ld = pd.DataFrame(self.xcorr)
        strong_ld_count = ((np.absolute(ld) > cutoff1) * ld).sum(axis = 0).sort_values(ascending = False)
        strong_ld_count = strong_ld_count[strong_ld_count > cutoff2]
        print('Filter by LD')
        exclude = []
        for x in strong_ld_count.index:
            if x in exclude:
                continue
            for y in strong_ld_count.index:
                if y in exclude or y == x:
                    continue
                if np.absolute(ld[x][y]) > cutoff3:
                    exclude.append(y)
        print('Done')
        out = [x for x in strong_ld_count.index if not x in exclude]
        return out

    def swap_B(self, top_set):
        '''
        Reorder rows in B so that strongest B appears in the specified "top_set" (set of indices)
        - useful when used with "select_convoluted_snps" to ensure the true effects are separated in different LD blocks
        - useful when simulating with annotations -- that at least for example indices near TSS will have strongest effects
        '''
        nb = np.zeros(self.B.shape)
        beta_max = np.amax(np.absolute(self.B), axis = 1)
        big_beta_index = [i[0] for i in sorted(enumerate(beta_max), key = lambda x: x[1], reverse = True)]
        for item in top_set:
            nb[item,:] = self.B[big_beta_index.pop(0),:]
        for idx in range(nb.shape[0]):
            if not idx in top_set:
                nb[idx,:] = self.B[big_beta_index.pop(0),:]
        self.B = nb
