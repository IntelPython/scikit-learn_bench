#===============================================================================
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench
import numpy as np
from daal4py import pca, pca_transform, normalization_zscore
from daal4py.sklearn._utils import getFPType

from sklearn.utils.extmath import svd_flip


parser = argparse.ArgumentParser(description='daal4py PCA benchmark')
parser.add_argument('--svd-solver', type=str,
                    choices=['daal', 'full', 'correlation'],
                    default='daal', help='SVD solver to use')
parser.add_argument('--n-components', type=int, default=None,
                    help='Number of components to find')
parser.add_argument('--whiten', action='store_true', default=False,
                    help='Perform whitening')
parser.add_argument('--write-results', action='store_true', default=False,
                    help='Write results to disk for verification')
params = bench.parse_args(parser)

# Load data
X_train, X_test, _, _ = bench.load_data(params, generated_data=['X_train'],
                                        add_dtype=True)

if params.n_components is None:
    p, n = X_train.shape
    params.n_components = min((n, (2 + min((n, p))) // 3))


# Define how to do our scikit-learn PCA using DAAL...
def pca_fit_daal(X, n_components, method):

    if n_components < 1:
        n_components = min(X.shape)

    fptype = getFPType(X)

    centering_algo = normalization_zscore(
        fptype=fptype,
        doScale=False
    )

    pca_algorithm = pca(
        fptype=fptype,
        method=method,
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

    fit_result, eigenvalues, eigenvectors, S = pca_fit_daal(X, min(X.shape), 'svdDense')
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
        method = 'correlationDense' if params.svd_solver == 'correlation' else 'svdDense'
        return pca_fit_daal(X, params.n_components, method)


def test_transform(Xp, pca_result, eigenvalues, eigenvectors):
    return pca_transform_daal(pca_result, Xp, params.n_components,
                              X_train.shape[0], eigenvalues,
                              eigenvectors, whiten=params.whiten)


# Time fit
fit_time, res = bench.measure_function_time(test_fit, X_train, params=params)

# Time transform
transform_time, tr = bench.measure_function_time(
    test_transform, X_test, *res[:3], params=params)

bench.print_output(library='daal4py', algorithm='pca',
                   stages=['training', 'transformation'],
                   params=params, functions=['PCA.fit', 'PCA.transform'],
                   times=[fit_time, transform_time], accuracy_type=None,
                   accuracies=[None, None], data=[X_train, X_test],
                   alg_params={'svd_solver': params.svd_solver,
                               'n_components': params.n_components})
