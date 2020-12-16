#===============================================================================
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench

import daal4py
from daal4py.sklearn._utils import getFPType


def compute_distances(pairwise_distances, X):
    algorithm = pairwise_distances(fptype=getFPType(X))
    return algorithm.compute(X)


parser = argparse.ArgumentParser(description='daal4py pairwise distances '
                                             'benchmark')
parser.add_argument('--metric', default='cosine',
                    choices=['cosine', 'correlation'],
                    help='Metric to test for pairwise distances')
params = bench.parse_args(parser)

# Load data
X, _, _, _ = bench.load_data(params, generated_data=['X_train'], add_dtype=True)

pairwise_distances = getattr(daal4py, f'{params.metric}_distance')

time, _ = bench.measure_function_time(
    compute_distances, pairwise_distances, X, params=params)

bench.print_output(library='daal4py', algorithm='distances', stages=['computation'],
                   params=params, functions=[params.metric.capitalize()], times=[time],
                   accuracy_type=None, accuracies=[None], data=[X],
                   alg_params={'metric': params.metric})
