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

from cuml import DBSCAN
from sklearn.metrics.cluster import davies_bouldin_score

parser = argparse.ArgumentParser(description='cuML DBSCAN benchmark')
parser.add_argument('-e', '--eps', '--epsilon', type=float, default=10.,
                    help='Radius of neighborhood of a point')
parser.add_argument('-m', '--min-samples', default=5, type=int,
                    help='The minimum number of samples required in a '
                    'neighborhood to consider a point a core point')
params = bench.parse_args(parser)

# Load generated data
X, _, _, _ = bench.load_data(params)

# Create our clustering object
dbscan = DBSCAN(eps=params.eps,
                min_samples=params.min_samples)

# Time fit
time, _ = bench.measure_function_time(dbscan.fit, X, params=params)
labels = dbscan.labels_

X_host = bench.convert_to_numpy(X)
labels_host = bench.convert_to_numpy(labels)

acc = davies_bouldin_score(X_host, labels_host)
params.n_clusters = len(set(labels_host)) - (1 if -1 in labels_host else 0)

bench.print_output(library='cuml', algorithm='dbscan', stages=['training'],
                   params=params, functions=['DBSCAN'], times=[time],
                   accuracies=[acc], accuracy_type='davies_bouldin_score', data=[X],
                   alg_instance=dbscan)
