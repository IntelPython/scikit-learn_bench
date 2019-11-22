# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row
import daal4py
from daal4py.sklearn.utils import getFPType
import numpy as np

parser = argparse.ArgumentParser(description='daal4py pairwise distances '
                                             'benchmark')
parser.add_argument('--metrics', nargs='*', default=['cosine', 'correlation'],
                    choices=('cosine', 'correlation'),
                    help='Metrics to test for pairwise_distances')
params = parse_args(parser, size=(1000, 150000), dtypes=('f8', 'f4'),
                    prefix='daal4py')

# Generate random data
X = np.random.rand(*params.shape).astype(params.dtype)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'time')
print_header(columns, params)

for metric in params.metrics:
    pairwise_distances = getattr(daal4py, f'{metric}_distance')

    def test_distances(pairwise_distances, X):
        algorithm = pairwise_distances(fptype=getFPType(X))
        return algorithm.compute(X)

    time, _ = time_mean_min(test_distances, pairwise_distances, X,
                            outer_loops=params.outer_loops,
                            inner_loops=params.inner_loops,
                            goal_outer_loops=params.goal,
                            time_limit=params.time_limit,
                            verbose=params.verbose)
    print_row(columns, params, function=metric.capitalize(), time=time)
