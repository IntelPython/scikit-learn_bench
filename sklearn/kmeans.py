# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row,\
    size_str, convert_data, get_dtype
import numpy as np
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

# Load and convert generated data
X = convert_data(np.load(params.filex),
                 params.dtype, params.data_order, params.data_format)
X_init = np.load(params.filei)

params.n_clusters = X_init.shape[0]

# Create our clustering object
kmeans = KMeans(n_clusters=params.n_clusters, n_jobs=params.n_jobs,
                tol=params.tol, max_iter=params.maxiter, n_init=1, init=X_init)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'n_clusters', 'time')
params.dtype = get_dtype(X)
params.size = size_str(X.shape)

# Time fit
fit_time, _ = time_mean_min(kmeans.fit, X,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)

# Time predict
predict_time, _ = time_mean_min(kmeans.predict, X,
                                outer_loops=params.predict_outer_loops,
                                inner_loops=params.predict_inner_loops,
                                goal_outer_loops=params.predict_goal,
                                time_limit=params.predict_time_limit,
                                verbose=params.verbose)

if params.output_format == "csv":
    print_header(columns, params)
    print_row(columns, params, function='KMeans.fit', time=fit_time)
    print_row(columns, params, function='KMeans.predict', time=predict_time)
elif params.output_format == "json":
    import json

    res = {
        "lib": "sklearn",
        "algorithm": "kmeans",
        "stage": "training",
        "data_format": params.data_format,
        "data_type": str(params.dtype),
        "data_order": params.data_order,
        "rows": X.shape[0],
        "columns": X.shape[1],
        "n_clusters": params.n_clusters,
        "time[s]": fit_time,
        "inertia": kmeans.inertia_,
        "algorithm_paramaters": dict(kmeans.get_params())
    }

    print(json.dumps(res, indent=4))

    res.update({
        "stage": "prediction",
        "time[s]": predict_time
    })

    print(json.dumps(res, indent=4))
