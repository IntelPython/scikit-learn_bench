# Copyright (C) 2017-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, measure_function_time, print_output, load_data,
    import_fptype_getter
)
import daal4py
getFPType = import_fptype_getter()


def compute_distances(pairwise_distances, X):
    algorithm = pairwise_distances(fptype=getFPType(X))
    return algorithm.compute(X)


parser = argparse.ArgumentParser(description='daal4py pairwise distances '
                                             'benchmark')
parser.add_argument('--metric', default='cosine',
                    choices=['cosine', 'correlation'],
                    help='Metric to test for pairwise distances')
params = parse_args(parser, size=(1000, 150000))

# Load data
X, _, _, _ = load_data(params, generated_data=['X_train'], add_dtype=True)

pairwise_distances = getattr(daal4py, f'{params.metric}_distance')

time, _ = measure_function_time(
    compute_distances, pairwise_distances, X, params=params)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'time')

print_output(library='daal4py', algorithm='distances', stages=['computation'],
             columns=columns, params=params,
             functions=[params.metric.capitalize()], times=[time],
             accuracy_type=None, accuracies=[None], data=[X],
             alg_params={'metric': params.metric})
