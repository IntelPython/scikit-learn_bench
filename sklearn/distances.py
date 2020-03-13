# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

parser = argparse.ArgumentParser(description='scikit-learn pairwise distances '
                                             'benchmark')
parser.add_argument('--metrics', nargs='*', default=['cosine', 'correlation'],
                    help='Metrics to test for pairwise_distances')
params = parse_args(parser, size=(1000, 150000), dtypes=('f8', 'f4'),
                    n_jobs_supported=True)

# Generate random data
X = np.random.rand(*params.shape).astype(params.dtype)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'time')
print_header(columns, params)

for metric in params.metrics:
    time, _ = time_mean_min(pairwise_distances, X, metric=metric,
                            n_jobs=params.n_jobs,
                            outer_loops=params.outer_loops,
                            inner_loops=params.inner_loops,
                            goal_outer_loops=params.goal,
                            time_limit=params.time_limit,
                            verbose=params.verbose)
    print_row(columns, params, function=metric.capitalize(), time=time)
