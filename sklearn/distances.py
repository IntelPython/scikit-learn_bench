# Copyright (C) 2017-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, measure_function_time, print_header, print_row, load_data,
    gen_basic_dict
)
from sklearn.metrics.pairwise import pairwise_distances

parser = argparse.ArgumentParser(description='scikit-learn pairwise distances '
                                             'benchmark')
parser.add_argument('--metrics', nargs='*', default=['cosine', 'correlation'],
                    help='Metrics to test for pairwise_distances')
params = parse_args(parser, size=(1000, 150000))

# Generate and convert random data
X, _, _, _ = load_data(params, generated_data=['X_train'])

# workaround for 'dtype' property absense in pandas DataFrame
if params.data_format == 'pandas':
    X.dtype = X.values.dtype

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'time')

if params.output_format == 'csv':
    print_header(columns, params)

for metric in params.metrics:
    time, _ = measure_function_time(pairwise_distances, X, metric=metric,
                                    n_jobs=params.n_jobs, params=params)
    if params.output_format == 'csv':
        print_row(columns, params, function=metric.capitalize(), time=time)
    elif params.output_format == 'json':
        import json

        result = gen_basic_dict('sklearn', 'distances', 'computation',
                                params, X)

        result.update({
            'metric': metric,
            'time[s]': time
        })

        print(json.dumps(result, indent=4),
              end=',\n' if metric != params.metrics[-1] else '\n')
