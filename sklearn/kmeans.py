# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, time_mean_min, load_data, gen_basic_dict, output_csv
)
import numpy as np
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='scikit-learn K-means benchmark')
parser.add_argument('-i', '--filei', '--fileI', '--init',
                    type=str, help='Initial clusters')
parser.add_argument('-t', '--tol', type=float, default=0.,
                    help='Absolute threshold')
parser.add_argument('--maxiter', type=int, default=100,
                    help='Maximum number of iterations')
parser.add_argument('--n-clusters', type=int, help='Number of clusters')
params = parse_args(parser, loop_types=('fit', 'predict'))

# Load and convert generated data
X_train, X_test, _, _ = load_data(params)

if params.filei is not None:
    X_init = np.load(params.filei).astype(params.dtype)
    params.n_clusters = X_init.shape[0]
else:
    np.random.seed(params.seed)
    centroids_idx = np.random.randint(0, X_train.shape[0],
                                      size=params.n_clusters)
    if hasattr(X_train, "iloc"):
        X_init = X_train.iloc[centroids_idx].values
    else:
        X_init = X_train[centroids_idx]

# Create our clustering object
kmeans = KMeans(n_clusters=params.n_clusters, n_jobs=params.n_jobs,
                tol=params.tol, max_iter=params.maxiter, n_init=1, init=X_init)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'n_clusters', 'time')

# Time fit
fit_time, _ = time_mean_min(kmeans.fit, X_train,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)
train_inertia = kmeans.inertia_

# Time predict
predict_time, _ = time_mean_min(kmeans.predict, X_test,
                                outer_loops=params.predict_outer_loops,
                                inner_loops=params.predict_inner_loops,
                                goal_outer_loops=params.predict_goal,
                                time_limit=params.predict_time_limit,
                                verbose=params.verbose)
test_inertia = kmeans.inertia_

if params.output_format == 'csv':
    output_csv(columns, params, functions=['KMeans.fit', 'KMeans.predict'],
               times=[fit_time, predict_time])
elif params.output_format == 'json':
    import json

    result = gen_basic_dict('sklearn', 'kmeans', 'training', params,
                            X_train, kmeans)
    result.update({
        'n_clusters': params.n_clusters,
        'time[s]': fit_time,
        'inertia': train_inertia
    })
    if 'init' in result['algorithm_parameters'].keys():
        if not isinstance(result['algorithm_parameters']['init'], str):
            result['algorithm_parameters']['init'] = 'random'
    print(json.dumps(result, indent=4))

    result = gen_basic_dict('sklearn', 'kmeans', 'prediction', params,
                            X_test, kmeans)
    result.update({
        'n_clusters': params.n_clusters,
        'time[s]': predict_time,
        'inertia': test_inertia
    })
    if 'init' in result['algorithm_parameters'].keys():
        if not isinstance(result['algorithm_parameters']['init'], str):
            result['algorithm_parameters']['init'] = 'random'
    print(json.dumps(result, indent=4))
