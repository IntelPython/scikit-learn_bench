# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row, size_str
from daal4py import kmeans
import numpy as np

parser = argparse.ArgumentParser(description='daal4py K-Means clustering '
                                             'benchmark')
parser.add_argument('-x', '--filex', '--fileX', '--input',
                    type=str, help='Points to cluster')
parser.add_argument('-i', '--filei', '--fileI', '--init',
                    type=str, help='Initial clusters')
# parser.add_argument('-t', '--filet', '--fileT', '--tol',
#                     type=str, help='Absolute threshold')
parser.add_argument('-m', '--data-multiplier', default=100,
                    type=int, help='Data multiplier')
parser.add_argument('--maxiter', type=int, default=100,
                    help='Maximum number of iterations')
params = parse_args(parser, loop_types=('fit', 'predict'), prefix='daal4py')

# Load generated data
X = np.load(params.filex)
X_init = np.load(params.filei)
X_mult = np.vstack((X,) * params.data_multiplier)

params.size = size_str(X.shape)
params.n_clusters = X_init.shape[0]
params.dtype = X.dtype


# Define functions to time
def test_fit(X, X_init):
    algorithm = kmeans(
        nClusters=params.n_clusters,
        maxIterations=params.maxiter
    )  # FIXME tolerance?
    return algorithm.compute(X, X_init)


def test_predict(X, X_init):
    algorithm = kmeans(
        nClusters=params.n_clusters,
        maxIterations=0
    )  # FIXME tolerance
    return algorithm.compute(X, X_init)


columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'n_clusters', 'time')
print_header(columns, params)

# Time fit
fit_time, _ = time_mean_min(test_fit, X, X_init,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops)
print_row(columns, params, function='KMeans.fit', time=fit_time)

# Time predict
predict_time, _ = time_mean_min(test_predict, X, X_init,
                                outer_loops=params.predict_outer_loops,
                                inner_loops=params.predict_inner_loops)
print_row(columns, params, function='KMeans.predict', time=predict_time)
