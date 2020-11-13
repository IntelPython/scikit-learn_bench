# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, measure_function_time, load_data, print_output, convert_to_numpy
from cuml import DBSCAN
from sklearn.metrics.cluster import davies_bouldin_score

parser = argparse.ArgumentParser(description='cuML DBSCAN benchmark')
parser.add_argument('-e', '--eps', '--epsilon', type=float, default=10.,
                    help='Radius of neighborhood of a point')
parser.add_argument('-m', '--min-samples', default=5, type=int,
                    help='The minimum number of samples required in a '
                    'neighborhood to consider a point a core point')
params = parse_args(parser)

# Load generated data
X, _, _, _ = load_data(params)

# Create our clustering object
dbscan = DBSCAN(eps=params.eps,
                min_samples=params.min_samples)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'n_clusters', 'time')

# Time fit
time, _ = measure_function_time(dbscan.fit, X, params=params)
labels = dbscan.labels_

X_host = convert_to_numpy(X)
labels_host = convert_to_numpy(labels)

acc = davies_bouldin_score(X_host, labels_host)
params.n_clusters = len(set(labels_host)) - (1 if -1 in labels_host else 0)

print_output(library='cuml', algorithm='dbscan', stages=['training'],
             columns=columns, params=params, functions=['DBSCAN'],
             times=[time], accuracies=[acc], accuracy_type='davies_bouldin_score', data=[X],
             alg_instance=dbscan)
