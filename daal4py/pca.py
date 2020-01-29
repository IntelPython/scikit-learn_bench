# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, time_mean_min, output_csv, load_data, gen_basic_dict
)
import numpy as np
from daal4py import pca, pca_transform, normalization_zscore
from daal4py.sklearn.utils import getFPType
from sklearn.utils.extmath import svd_flip

parser = argparse.ArgumentParser(description='daal4py PCA benchmark')
parser.add_argument('--svd-solver', type=str, choices=['daal', 'full'],
                    default='daal', help='SVD solver to use')
parser.add_argument('--n-components', type=int, default=None,
                    help='Number of components to find')
parser.add_argument('--whiten', action='store_true', default=False,
                    help='Perform whitening')
parser.add_argument('--write-results', action='store_true', default=False,
                    help='Write results to disk for verification')
params = parse_args(parser, size=(10000, 1000),
                    loop_types=('fit', 'transform'))

# Load data
X_train, X_test, _, _ = load_data(params, generated_data=['X_train', 'X_test'],
                                  add_dtype=True)

if not params.n_components:
    p, n = X_train.shape
    params.n_components = min((n, (2 + min((n, p))) // 3))


# Define how to do our scikit-learn PCA using DAAL...
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
            tr_data['eigenvalue'] = (fit_n_samples - 1) \
                * pca_result.eigenvalues
        else:
            tr_data['eigenvalue'] = pca_result.eigenvalues
    elif scale_eigenvalues:
        tr_data['eigenvalue'] = np.full((1, pca_result.eigenvalues.size),
                                        fit_n_samples - 1, dtype=X.dtype)

    transform_algorithm = pca_transform(
        fptype=fptype,
        nComponents=n_components
    )
    transform_result = transform_algorithm.compute(X, pca_result.eigenvectors,
                                                   tr_data)
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


def test_fit(X):
    if params.svd_solver == 'full':
        return pca_fit_full_daal(X, params.n_components)
    else:
        return pca_fit_daal(X, params.n_components)


def test_transform(Xp, pca_result, eigenvalues, eigenvectors):
    return pca_transform_daal(pca_result, Xp, params.n_components, X_train.shape[0],
                              eigenvalues, eigenvectors, whiten=params.whiten)


columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'svd_solver', 'n_components', 'whiten', 'time')

# Time fit
fit_time, res = time_mean_min(test_fit, X_train,
                              outer_loops=params.fit_outer_loops,
                              inner_loops=params.fit_inner_loops,
                              goal_outer_loops=params.fit_goal,
                              time_limit=params.fit_time_limit,
                              verbose=params.verbose)

# Time transform
transform_time, tr = time_mean_min(test_transform, X_test, *res[:3],
                                   outer_loops=params.transform_outer_loops,
                                   inner_loops=params.transform_inner_loops,
                                   goal_outer_loops=params.transform_goal,
                                   time_limit=params.transform_time_limit,
                                   verbose=params.verbose)

if params.output_format == 'csv':
    output_csv(columns, params, functions=['PCA.fit', 'PCA.transform'],
               times=[fit_time, transform_time])
elif params.output_format == 'json':
    import json

    result = gen_basic_dict('daal4py', 'pca', 'training', params, X_train)
    result.update({
        'time[s]': fit_time
    })
    print(json.dumps(result, indent=4))

    result = gen_basic_dict('daal4py', 'pca', 'transform', params, X_test)
    result.update({
        'time[s]': transform_time
    })
    print(json.dumps(result, indent=4))

if params.write_results:
    np.save('pca_daal4py_X.npy', X)
    np.save('pca_daal4py_Xp.npy', Xp)
    np.save('pca_daal4py_eigvals.npy', res[1])
    np.save('pca_daal4py_eigvecs.npy', res[2])
    np.save('pca_daal4py_tr.npy', tr)
