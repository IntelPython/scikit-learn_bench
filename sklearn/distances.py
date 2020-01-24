# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row,\
    convert_data, get_dtype
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

parser = argparse.ArgumentParser(description='scikit-learn pairwise distances '
                                             'benchmark')
parser.add_argument('--metrics', nargs='*', default=['cosine', 'correlation'],
                    help='Metrics to test for pairwise_distances')
params = parse_args(parser, size=(1000, 150000))

# Generate and convert random data
X = convert_data(np.random.rand(*params.shape),
                 params.dtype, params.data_order, params.data_format)

# workaround for "dtype" property absense in pandas DataFrame
if params.data_format == "pandas":
    X.dtype = X.values.dtype

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'time')
params.dtype = get_dtype(X)

if params.output_format == "csv":
    print_header(columns, params)

for metric in params.metrics:
    time, _ = time_mean_min(pairwise_distances, X, metric=metric,
                            n_jobs=params.n_jobs,
                            outer_loops=params.outer_loops,
                            inner_loops=params.inner_loops,
                            goal_outer_loops=params.goal,
                            time_limit=params.time_limit,
                            verbose=params.verbose)
    if params.output_format == "csv":
        print_row(columns, params, function=metric.capitalize(), time=time)
    elif params.output_format == "json":
        import json

        res = {
            "lib": "sklearn",
            "algorithm": "distances",
            "metric": metric,
            "data_format": params.data_format,
            "data_type": str(params.dtype),
            "data_order": params.data_order,
            "rows": X.shape[0],
            "columns": X.shape[1],
            "time[s]": time
        }

        print(json.dumps(res, indent=4))
