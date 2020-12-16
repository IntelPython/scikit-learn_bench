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

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import davies_bouldin_score

parser = argparse.ArgumentParser(description='scikit-learn K-means benchmark')
parser.add_argument('-i', '--filei', '--fileI', '--init',
                    type=str, help='Initial clusters')
parser.add_argument('-t', '--tol', type=float, default=0.,
                    help='Absolute threshold')
parser.add_argument('--maxiter', type=int, default=100,
                    help='Maximum number of iterations')
parser.add_argument('--n-clusters', type=int, help='Number of clusters')
params = bench.parse_args(parser)

# Load and convert generated data
X_train, X_test, _, _ = bench.load_data(params)

if params.filei == 'k-means++':
    X_init = 'k-means++'
# Load initial centroids from specified path
elif params.filei is not None:
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


def fit_kmeans(X):
    global X_init, params
    alg = KMeans(n_clusters=params.n_clusters, tol=params.tol,
                 max_iter=params.maxiter, init=X_init, n_init=1)
    alg.fit(X)
    return alg


# Time fit
fit_time, kmeans = bench.measure_function_time(fit_kmeans, X_train, params=params)

train_predict = kmeans.predict(X_train)
acc_train = davies_bouldin_score(X_train, train_predict)

# Time predict
predict_time, test_predict = bench.measure_function_time(
    kmeans.predict, X_test, params=params)

acc_test = davies_bouldin_score(X_test, test_predict)

bench.print_output(library='sklearn', algorithm='kmeans',
                   stages=['training', 'prediction'],
                   params=params, functions=['KMeans.fit', 'KMeans.predict'],
                   times=[fit_time, predict_time], accuracy_type='davies_bouldin_score',
                   accuracies=[acc_train, acc_test], data=[X_train, X_test],
                   alg_instance=kmeans)
