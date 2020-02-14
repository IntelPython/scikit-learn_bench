# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row, size_str
from daal4py import dbscan
from daal4py.sklearn.utils import getFPType
import numpy as np

parser = argparse.ArgumentParser(description='daal4py DBSCAN clustering '
                                             'benchmark')
parser.add_argument('-x', '--filex', '--fileX', '--input', required=True,
                    type=str, help='Points to cluster')
parser.add_argument('-e', '--eps', '--epsilon', type=float, default=10,
                    help='Radius of neighborhood of a point')
parser.add_argument('-m', '--data-multiplier', default=100,
                    type=int, help='Data multiplier')
parser.add_argument('-M', '--min-samples', default=5, type=int,
                    help='The minimum number of samples required in a '
                    'neighborhood to consider a point a core point')
params = parse_args(parser, prefix='daal4py')

# Load generated data
X = np.load(params.filex)
X_mult = np.vstack((X,) * params.data_multiplier)

params.size = size_str(X.shape)
params.dtype = X.dtype


# Define functions to time
def test_dbscan(X):
    algorithm = dbscan(
        fptype=getFPType(X),
        epsilon=params.eps,
        minObservations=params.min_samples,
        resultsToCompute='computeCoreIndices'
    )
    return algorithm.compute(X)


columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'n_clusters', 'time')
print_header(columns, params)

# Time clustering
time, result = time_mean_min(test_dbscan, X,
                             outer_loops=params.outer_loops,
                             inner_loops=params.inner_loops,
                             goal_outer_loops=params.goal,
                             time_limit=params.time_limit,
                             verbose=params.verbose)
params.n_clusters = result.nClusters[0, 0]
print_row(columns, params, function='DBSCAN', time=time)
