# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, load_data, print_output
from sklearn.cluster import DBSCAN

parser = argparse.ArgumentParser(description='scikit-learn DBSCAN benchmark')
parser.add_argument('-e', '--eps', '--epsilon', type=float, default=10.,
                    help='Radius of neighborhood of a point')
parser.add_argument('-m', '--min-samples', default=5, type=int,
                    help='The minimum number of samples required in a '
                    'neighborhood to consider a point a core point')
params = parse_args(parser, n_jobs_supported=True)

# Load generated data
X, _, _, _ = load_data(params, add_dtype=True)

# Create our clustering object
dbscan = DBSCAN(eps=params.eps, n_jobs=params.n_jobs,
                min_samples=params.min_samples, metric='euclidean',
                algorithm='auto')

# N.B. algorithm='auto' will select DAAL's brute force method when running
# daal4py-patched scikit-learn, and probably 'kdtree' when running unpatched
# scikit-learn.

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'n_clusters', 'time')

# Time fit
time, _ = time_mean_min(dbscan.fit, X,
                        outer_loops=params.outer_loops,
                        inner_loops=params.inner_loops,
                        goal_outer_loops=params.goal,
                        time_limit=params.time_limit,
                        verbose=params.verbose)
labels = dbscan.labels_
params.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print_output(library='sklearn', algorithm='dbscan', stages=['training'],
             columns=columns, params=params, functions=['DBSCAN'],
             times=[time], accuracies=[None], accuracy_type=None, data=[X])
