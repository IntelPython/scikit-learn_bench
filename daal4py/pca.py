# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import print_function
import numpy as np
import timeit
from numpy.random import rand
from daal4py import pca, pca_transform, normalization_zscore
from daal4py.sklearn.utils import getFPType
from sklearn.utils.extmath import svd_flip
from args import getArguments, coreString
from bench import prepare_benchmark

import argparse
parser = argparse.ArgumentParser(description='daal4py PCA benchmark')
parser.add_argument('--svd-solver', type=str, choices=['daal', 'full'],
                    default='daal', help='SVD solver to use')
parser.add_argument('--n-components', type=int, default=None,
                    help='Number of components to find')
parser.add_argument('--whiten', action='store_true', default=False,
                    help='Perform whitening')
parser.add_argument('--write-results', action='store_true', default=False,
                    help='Write results to disk for verification')
args = getArguments(parser)
REP = args.iteration if args.iteration != '?' else 10
core_number, daal_version = prepare_benchmark(args)


def st_time(func):
    def st_func(*args, **keyArgs):
        times = []
        for n in range(REP):
            t1 = timeit.default_timer()
            r = func(*args, **keyArgs)
            t2 = timeit.default_timer()
            times.append(t2-t1)
        print(min(times))
        return r
    return st_func

p = args.size[0]
n = args.size[1]
X = rand(p,n)
Xp = rand(p,n)

if args.n_components:
    n_components = args.n_components
else:
    n_components = min((n, (2 + min((n, p))) // 3))

def pca_fit_daal(X, n_components):

    if n_components < 1:
        n_components = min(X.shape)

    fptype = getFPType(X)

    centering_algo = normalization_zscore(
            fptype=fptype,
            doScale=False
    )

    pca_algorithm = pca(
            fptype=fptype,
            method='svdDense',
            normalization=centering_algo,
            resultsToCompute='mean|variance|eigenvalue',
            isDeterministic=True,
            nComponents=n_components
    )

    pca_result = pca_algorithm.compute(X)
    eigenvectors = pca_result.eigenvectors
    eigenvalues = pca_result.eigenvalues.ravel()
    singular_values = np.sqrt((X.shape[0] - 1) * eigenvalues)

    return pca_result, eigenvalues, eigenvectors, singular_values


def pca_transform_daal(pca_result, X, n_components, fit_n_samples,
                       eigenvalues, eigenvectors,
                       whiten=False, scale_eigenvalues=False):

    fptype = getFPType(X)

    tr_data = {}
    tr_data['mean'] = pca_result.dataForTransform['mean']

    if whiten:
        if scale_eigenvalues:
            tr_data['eigenvalue'] = (fit_n_samples - 1) * pca_result.eigenvalues
        else:
            tr_data['eigenvalue'] = pca_result.eigenvalues
    elif scale_eigenvalues:
        tr_data['eigenvalue'] = np.full((1, pca_result.eigenvalues.size),
                                        fit_n_samples - 1, dtype=X.dtype)

    transform_algorithm = pca_transform(fptype=fptype, nComponents=n_components)
    transform_result = transform_algorithm.compute(X, pca_result.eigenvectors, tr_data)
    return transform_result.transformedData


def pca_fit_full_daal(X, n_components):

    fit_result, eigenvalues, eigenvectors, S = pca_fit_daal(X, min(X.shape))
    U = pca_transform_daal(fit_result, X, min(X.shape), X.shape[0],
                           eigenvalues, eigenvectors,
                           whiten=True, scale_eigenvalues=True)
    V = fit_result.eigenvectors

    U, V = svd_flip(U, V)

    eigenvalues = fit_result.eigenvalues[:n_components].copy()
    eigenvectors = fit_result.eigenvectors[:n_components].copy()

    return fit_result, eigenvalues, eigenvectors, U, S, V


@st_time
def test_fit(X):
    if args.svd_solver == 'full':
        return pca_fit_full_daal(X, n_components)
    else:
        return pca_fit_daal(X, n_components)

@st_time
def test_transform(Xp, pca_result, eigenvalues, eigenvectors):
    return pca_transform_daal(pca_result, Xp, n_components, X.shape[0],
                              eigenvalues, eigenvectors,
                              whiten=args.whiten)


print (','.join([args.batchID, args.arch, args.prefix, "PCA.fit", coreString(args.num_threads), "Double", "%sx%s" % (p,n)]), end=',')
res = test_fit(X)
print (','.join([args.batchID, args.arch, args.prefix, "PCA.transform", coreString(args.num_threads), "Double", "%sx%s" % (p,n)]), end=',')
tr = test_transform(Xp, res[0], res[1], res[2])


if args.write_results:
    np.save('pca_daal4py_X.npy', X)
    np.save('pca_daal4py_Xp.npy', Xp)
    np.save('pca_daal4py_eigvals.npy', res[1])
    np.save('pca_daal4py_eigvecs.npy', res[2])
    np.save('pca_daal4py_tr.npy', tr)
