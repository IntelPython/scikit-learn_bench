# Copyright (C) 2017-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, time_mean_min, load_data, print_output
)
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='scikit-learn PCA benchmark')
parser.add_argument('--svd-solver', type=str, choices=['daal', 'full'],
                    default='daal', help='SVD solver to use')
parser.add_argument('--n-components', type=int, default=None,
                    help='Number of components to find')
parser.add_argument('--whiten', action='store_true', default=False,
                    help='Perform whitening')
params = parse_args(parser, size=(10000, 1000),
                    loop_types=('fit', 'transform'))

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
fit_time, _ = time_mean_min(pca.fit, X_train,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)

# Time transform
transform_time, _ = time_mean_min(pca.transform, X_train,
                                  outer_loops=params.transform_outer_loops,
                                  inner_loops=params.transform_inner_loops,
                                  goal_outer_loops=params.transform_goal,
                                  time_limit=params.transform_time_limit,
                                  verbose=params.verbose)

print_output(library='sklearn', algorithm='pca',
             stages=['training', 'transformation'], columns=columns,
             params=params, functions=['PCA.fit', 'PCA.transform'],
             times=[fit_time, transform_time], accuracy_type=None,
             accuracies=[None, None], data=[X_train, X_test],
             alg_instance=pca)
