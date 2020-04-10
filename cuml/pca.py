# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, measure_function_time, load_data, print_output
)
from cuml import PCA

parser = argparse.ArgumentParser(description='cuML PCA benchmark')
parser.add_argument('--svd-solver', type=str, default='full',
                    choices=['auto', 'full', 'jacobi'],
                    help='SVD solver to use')
parser.add_argument('--n-components', type=int, default=None,
                    help='Number of components to find')
parser.add_argument('--whiten', action='store_true', default=False,
                    help='Perform whitening')
params = parse_args(parser, size=(10000, 1000))

# Load random data
X_train, X_test, _, _ = load_data(params, generated_data=['X_train'])

if params.n_components is None:
    p, n = X_train.shape
    params.n_components = min((n, (2 + min((n, p))) // 3))

# Create our PCA object
pca = PCA(svd_solver=params.svd_solver, whiten=params.whiten,
          n_components=params.n_components)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'svd_solver', 'n_components', 'whiten', 'time')

# Time fit
fit_time, _ = measure_function_time(pca.fit, X_train, params=params)

# Time transform
transform_time, _ = measure_function_time(
    pca.transform, X_train, params=params)

print_output(library='cuml', algorithm='pca',
             stages=['training', 'transformation'], columns=columns,
             params=params, functions=['PCA.fit', 'PCA.transform'],
             times=[fit_time, transform_time], accuracy_type=None,
             accuracies=[None, None], data=[X_train, X_test],
             alg_instance=pca)
