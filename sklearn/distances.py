# Copyright (C) 2017-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, measure_function_time, print_output, load_data
)
from sklearn.metrics.pairwise import pairwise_distances

parser = argparse.ArgumentParser(description='scikit-learn pairwise distances '
                                             'benchmark')
parser.add_argument('--metric', default='cosine',
                    choices=['cosine', 'correlation'],
                    help='Metric to test for pairwise distances')
params = parse_args(parser, size=(1000, 150000))

# Load data
X, _, _, _ = load_data(params, generated_data=['X_train'], add_dtype=True)

time, _ = measure_function_time(pairwise_distances, X, metric=params.metric,
                                n_jobs=params.n_jobs, params=params)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'time')

print_output(library='sklearn', algorithm='distances', stages=['computation'],
             columns=columns, params=params,
             functions=[params.metric.capitalize()], times=[time],
             accuracy_type=None, accuracies=[None], data=[X],
             alg_params={'metric': params.metric})
