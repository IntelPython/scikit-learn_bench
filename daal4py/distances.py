# Copyright (C) 2017-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, time_mean_min, print_header, print_row, load_data,
    gen_basic_dict
)
import daal4py
from daal4py.sklearn.utils import getFPType

parser = argparse.ArgumentParser(description='daal4py pairwise distances '
                                             'benchmark')
parser.add_argument('--metrics', nargs='*', default=['cosine', 'correlation'],
                    choices=('cosine', 'correlation'),
                    help='Metrics to test for pairwise_distances')
params = parse_args(parser, size=(1000, 150000), prefix='daal4py')

# Generate random data
X, _, _, _ = load_data(params, generated_data=['X_train'], add_dtype=True)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'time')

if params.output_format == 'csv':
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

    if params.output_format == 'csv':
        print_row(columns, params, function=metric.capitalize(), time=time)
    elif params.output_format == 'json':
        import json

        result = gen_basic_dict('daal4py', 'distances', 'computation',
                                params, X)

        result.update({
            'metric': metric,
            'time[s]': time
        })

        print(json.dumps(result, indent=4))
