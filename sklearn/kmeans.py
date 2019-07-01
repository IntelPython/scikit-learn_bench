# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row, size_str
import numpy as np
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='scikit-learn K-means benchmark')
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
params = parse_args(parser, loop_types=('fit', 'predict'))

# Load generated data
X = np.load(params.filex)
X_init = np.load(params.filei)
X_mult = np.vstack((X,) * params.data_multiplier)

n_clusters = X_init.shape[0]

# Create our clustering object
kmeans = KMeans(n_clusters=n_clusters, n_jobs=params.n_jobs, tol=1e-16,
                max_iter=params.maxiter, n_init=1, init=X_init)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'n_clusters', 'time')
params.size = size_str(X.shape)
params.n_clusters = n_clusters
params.dtype = X.dtype
print_header(columns, params)

# Time fit
fit_time, _ = time_mean_min(kmeans.fit, X,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops)
print_row(columns, params, function='KMeans.fit', time=fit_time)

# Time predict
predict_time, _ = time_mean_min(kmeans.predict, X,
                                outer_loops=params.predict_outer_loops,
                                inner_loops=params.predict_inner_loops)
print_row(columns, params, function='KMeans.predict', time=predict_time)
