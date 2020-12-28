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

from daal4py import dbscan
from daal4py.sklearn._utils import getFPType

parser = argparse.ArgumentParser(description='daal4py DBSCAN clustering '
                                             'benchmark')
parser.add_argument('-e', '--eps', '--epsilon', type=float, default=10.,
                    help='Radius of neighborhood of a point')
parser.add_argument('-m', '--min-samples', default=5, type=int,
                    help='The minimum number of samples required in a '
                    'neighborhood to consider a point a core point')
params = bench.parse_args(parser, prefix='daal4py')

# Load generated data
X, _, _, _ = bench.load_data(params, add_dtype=True)


# Define functions to time
def test_dbscan(X):
    algorithm = dbscan(
        fptype=getFPType(X),
        epsilon=params.eps,
        minObservations=params.min_samples,
        resultsToCompute='computeCoreIndices'
    )
    return algorithm.compute(X)


# Time clustering
time, result = bench.measure_function_time(test_dbscan, X, params=params)
params.n_clusters = int(result.nClusters[0, 0])

bench.print_output(library='daal4py', algorithm='dbscan', stages=['training'],
                   params=params, functions=['DBSCAN'], times=[time],
                   accuracies=[None], accuracy_type=None, data=[X])
