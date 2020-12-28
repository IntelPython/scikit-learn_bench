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

import numpy as np
from daal4py import kmeans
from daal4py.sklearn._utils import getFPType

parser = argparse.ArgumentParser(description='daal4py K-Means clustering '
                                             'benchmark')
parser.add_argument('-i', '--filei', '--fileI', '--init',
                    type=str, help='Initial clusters')
parser.add_argument('-t', '--tol', default=0., type=float,
                    help='Absolute threshold')
parser.add_argument('--maxiter', type=int, default=100,
                    help='Maximum number of iterations')
parser.add_argument('--n-clusters', type=int, help='Number of clusters')
params = bench.parse_args(parser, prefix='daal4py')

# Load generated data
X_train, X_test, _, _ = bench.load_data(params, add_dtype=True)

# Load initial centroids from specified path
if params.filei is not None:
    X_init = np.load(params.filei).astype(params.dtype)
    params.n_clusters = X_init.shape[0]
# or choose random centroids from training data
else:
    np.random.seed(params.seed)
    centroids_idx = np.random.randint(0, X_train.shape[0],
                                      size=params.n_clusters)
    if hasattr(X_train, "iloc"):
        X_init = X_train.iloc[centroids_idx].values
    else:
        X_init = X_train[centroids_idx]


# Define functions to time
def test_fit(X, X_init):
    algorithm = kmeans(
        fptype=getFPType(X),
        nClusters=params.n_clusters,
        maxIterations=params.maxiter,
        assignFlag=True,
        accuracyThreshold=params.tol
    )
    return algorithm.compute(X, X_init)


def test_predict(X, X_init):
    algorithm = kmeans(
        fptype=getFPType(X),
        nClusters=params.n_clusters,
        maxIterations=0,
        assignFlag=True,
        accuracyThreshold=0.0
    )
    return algorithm.compute(X, X_init)


# Time fit
fit_time, res = bench.measure_function_time(test_fit, X_train, X_init, params=params)
train_inertia = float(res.objectiveFunction[0, 0])

# Time predict
predict_time, res = bench.measure_function_time(
    test_predict, X_test, X_init, params=params)
test_inertia = float(res.objectiveFunction[0, 0])

bench.print_output(library='daal4py', algorithm='kmeans',
                   stages=['training', 'prediction'],
                   params=params, functions=['KMeans.fit', 'KMeans.predict'],
                   times=[fit_time, predict_time], accuracy_type='inertia',
                   accuracies=[train_inertia, test_inertia], data=[X_train, X_test])
