# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, load_data, gen_basic_dict, output_csv
from daal4py import kmeans
from daal4py.sklearn.utils import getFPType
import numpy as np

parser = argparse.ArgumentParser(description='daal4py K-Means clustering '
                                             'benchmark')
parser.add_argument('-i', '--filei', '--fileI', '--init',
                    type=str, help='Initial clusters')
parser.add_argument('-t', '--tol', default=0., type=float,
                    help='Absolute threshold')
parser.add_argument('--maxiter', type=int, default=100,
                    help='Maximum number of iterations')
parser.add_argument('--n-clusters', type=int, help='Number of clusters')
params = parse_args(parser, loop_types=('fit', 'predict'), prefix='daal4py')

# Load generated data
X_train, X_test, _, _ = load_data(params, add_dtype=True)

if params.filei is not None:
    X_init = np.load(params.filei).astype(params.dtype)
    params.n_clusters = X_init.shape[0]
else:
    np.random.seed(params.seed)
    centroids_idx = np.random.randint(0, X_train.shape[0], size=params.n_clusters)
    X_init = data[centroids_idx]

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
fit_time, res = time_mean_min(test_fit, X_train, X_init,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)
train_inertia = res.goalFunction[0,0]

# Time predict
predict_time, res = time_mean_min(test_predict, X_test, X_init,
                                outer_loops=params.predict_outer_loops,
                                inner_loops=params.predict_inner_loops,
                                goal_outer_loops=params.predict_goal,
                                time_limit=params.predict_time_limit,
                                verbose=params.verbose)
test_inertia = res.goalFunction[0,0]

if params.output_format == "csv":
    output_csv(columns, params, functions=['KMeans.fit', 'KMeans.predict'],
               times=[fit_time, predict_time])
elif params.output_format == "json":
    import json

    result = gen_basic_dict("daal4py", "kmeans", "training", params,
                            X_train)
    result.update({
        "n_clusters": params.n_clusters,
        "time[s]": fit_time,
        "inertia": train_inertia
    })
    print(json.dumps(result, indent=4))

    result = gen_basic_dict("daal4py", "kmeans", "prediction", params,
                            X_test)
    result.update({
        "n_clusters": params.n_clusters,
        "time[s]": predict_time,
        "inertia": test_inertia
    })
    print(json.dumps(result, indent=4))
