# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row, size_str, convert_data
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='scikit-learn K-means benchmark')
parser.add_argument('-x', '--filex', '--fileX', '--input', required=True,
                    type=str, help='Points to cluster')
parser.add_argument('-i', '--filei', '--fileI', '--init', required=True,
                    type=str, help='Initial clusters')
parser.add_argument('-t', '--tol', type=float, default=0.,
                    help='Absolute threshold')
parser.add_argument('-m', '--data-multiplier', default=100,
                    type=int, help='Data multiplier')
parser.add_argument('--maxiter', type=int, default=100,
                    help='Maximum number of iterations')
params = parse_args(parser, loop_types=('fit', 'predict'))

# Load generated data
X = np.load(params.filex)
X_init = np.load(params.filei)
X_mult = np.vstack((X,) * params.data_multiplier)

X = convert_data(X, X.dtype, params.data_order, params.data_type)

n_clusters = X_init.shape[0]

# Create our clustering object
kmeans = KMeans(n_clusters=n_clusters, n_jobs=params.n_jobs, tol=params.tol,
                max_iter=params.maxiter, n_init=1, init=X_init)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'n_clusters', 'time')
params.n_clusters = n_clusters
if params.data_type is "pandas":
    params.size = size_str(X.values.shape)
    params.dtype = X.values.dtype
else:
    params.size = size_str(X.shape)
    params.dtype = X.dtype
print_header(columns, params)

# Time fit
fit_time, _ = time_mean_min(kmeans.fit, X,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)
print_row(columns, params, function='KMeans.fit', time=fit_time)

# Time predict
predict_time, _ = time_mean_min(kmeans.predict, X,
                                outer_loops=params.predict_outer_loops,
                                inner_loops=params.predict_inner_loops,
                                goal_outer_loops=params.predict_goal,
                                time_limit=params.predict_time_limit,
                                verbose=params.verbose)
print_row(columns, params, function='KMeans.predict', time=predict_time)
