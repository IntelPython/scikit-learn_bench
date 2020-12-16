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
from cuml import KMeans
import warnings
from sklearn.metrics.cluster import davies_bouldin_score

warnings.filterwarnings('ignore', category=FutureWarning)
parser = argparse.ArgumentParser(description='cuML K-means benchmark')
parser.add_argument('-i', '--filei', '--fileI', '--init',
                    type=str, help='Initial clusters')
parser.add_argument('-t', '--tol', type=float, default=0.,
                    help='Absolute threshold')
parser.add_argument('--maxiter', type=int, default=100,
                    help='Maximum number of iterations')
parser.add_argument('--samples-per-batch', type=int, default=32768,
                    help='Maximum number of iterations')
parser.add_argument('--n-clusters', type=int, help='Number of clusters')
params = bench.parse_args(parser, prefix='cuml', loop_types=('fit', 'predict'))

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
        X_init = X_train.iloc[centroids_idx].to_pandas().values
    else:
        X_init = X_train[centroids_idx]


# Workaround for cuML kmeans fail
# when second call of 'fit' method causes AttributeError
def kmeans_fit(X):
    alg = KMeans(n_clusters=params.n_clusters, tol=params.tol,
                 max_iter=params.maxiter, init=X_init,
                 max_samples_per_batch=params.samples_per_batch)
    alg.fit(X)
    return alg


# Time fit
fit_time, kmeans = bench.measure_function_time(kmeans_fit, X_train, params=params)
train_predict = kmeans.predict(X_train)

# Time predict
predict_time, test_predict = bench.measure_function_time(kmeans.predict, X_test,
                                                         params=params)

X_train_host = bench.convert_to_numpy(X_train)
train_predict_host = bench.convert_to_numpy(train_predict)
acc_train = davies_bouldin_score(X_train_host, train_predict_host)

X_test_host = bench.convert_to_numpy(X_test)
test_predict_host = bench.convert_to_numpy(test_predict)

acc_test = davies_bouldin_score(X_test_host, test_predict_host)

bench.print_output(library='cuml', algorithm='kmeans',
                   stages=['training', 'prediction'], params=params,
                   functions=['KMeans.fit', 'KMeans.predict'],
                   times=[fit_time, predict_time], accuracy_type='davies_bouldin_score',
                   accuracies=[acc_train, acc_test], data=[X_train, X_test],
                   alg_instance=kmeans)
