# Copyright (C) 2017-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, measure_function_time, load_data, print_output, getFPType
)
import numpy as np
from daal4py import kmeans


parser = argparse.ArgumentParser(description='daal4py K-Means clustering '
                                             'benchmark')
parser.add_argument('-i', '--filei', '--fileI', '--init',
                    type=str, help='Initial clusters')
parser.add_argument('-t', '--tol', default=0., type=float,
                    help='Absolute threshold')
parser.add_argument('--maxiter', type=int, default=100,
                    help='Maximum number of iterations')
parser.add_argument('--n-clusters', type=int, help='Number of clusters')
params = parse_args(parser, prefix='daal4py')

# Load generated data
X_train, X_test, _, _ = load_data(params, add_dtype=True)

# Load initial centroids from specified path
if params.filei is not None:
    X_init = np.load(params.filei).astype(params.dtype)
    params.n_clusters = X_init.shape[0]
# or choose random centroids from training data
else:
    np.random.seed(params.seed)
    centroids_idx = np.random.randint(0, X_train.shape[0],
                                      size=params.n_clusters)
    if hasattr(X_train, "iloc"):
        X_init = X_train.iloc[centroids_idx].values
    else:
        X_init = X_train[centroids_idx]


# Define functions to time
def test_fit(X, X_init):
    algorithm = kmeans(
        fptype=getFPType(X),
        nClusters=params.n_clusters,
        maxIterations=params.maxiter,
        assignFlag=True,
        accuracyThreshold=params.tol
    )
    return algorithm.compute(X, X_init)


def test_predict(X, X_init):
    algorithm = kmeans(
        fptype=getFPType(X),
        nClusters=params.n_clusters,
        maxIterations=0,
        assignFlag=True,
        accuracyThreshold=0.0
    )
    return algorithm.compute(X, X_init)


columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'n_clusters', 'time')

# Time fit
fit_time, res = measure_function_time(test_fit, X_train, X_init, params=params)
train_inertia = float(res.goalFunction[0, 0])

# Time predict
predict_time, res = measure_function_time(
    test_predict, X_test, X_init, params=params)
test_inertia = float(res.goalFunction[0, 0])

print_output(library='daal4py', algorithm='kmeans',
             stages=['training', 'prediction'], columns=columns,
             params=params, functions=['KMeans.fit', 'KMeans.predict'],
             times=[fit_time, predict_time], accuracy_type='inertia',
             accuracies=[train_inertia, test_inertia], data=[X_train, X_test])
