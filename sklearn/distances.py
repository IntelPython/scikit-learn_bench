# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row, convert_data
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

parser = argparse.ArgumentParser(description='scikit-learn pairwise distances '
                                             'benchmark')
parser.add_argument('--metrics', nargs='*', default=['cosine', 'correlation'],
                    help='Metrics to test for pairwise_distances')
params = parse_args(parser, size=(1000, 150000), dtypes=('f8', 'f4'))

# Generate random data
X = np.random.rand(*params.shape).astype(params.dtype)

X = convert_data(X, X.dtype, params.data_order, params.data_type)
if params.data_type is "pandas":
    X.dtype = X.values.dtype

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
