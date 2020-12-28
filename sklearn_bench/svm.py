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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description='scikit-learn SVM benchmark')

parser.add_argument('-C', dest='C', type=float, default=1.0,
                    help='SVM regularization parameter')
parser.add_argument('--kernel', choices=('linear', 'rbf'),
                    default='linear', help='SVM kernel function')
parser.add_argument('--gamma', type=float, default=None,
                    help='Parameter for kernel="rbf"')
parser.add_argument('--maxiter', type=int, default=-1,
                    help='Maximum iterations for the iterative solver. '
                         '-1 means no limit.')
parser.add_argument('--max-cache-size', type=int, default=8,
                    help='Maximum cache size, in gigabytes, for SVM.')
parser.add_argument('--tol', type=float, default=1e-3,
                    help='Tolerance passed to sklearn.svm.SVC')
parser.add_argument('--no-shrinking', action='store_false', default=True,
                    dest='shrinking', help="Don't use shrinking heuristic")
params = bench.parse_args(parser, loop_types=('fit', 'predict'))

# Load data
X_train, X_test, y_train, y_test = bench.load_data(params)

if params.gamma is None:
    params.gamma = 1.0 / X_train.shape[1]

cache_size_bytes = bench.get_optimal_cache_size(X_train.shape[0],
                                                max_cache=params.max_cache_size)
params.cache_size_mb = cache_size_bytes / 1024**2
params.n_classes = len(np.unique(y_train))

# Create our C-SVM classifier
clf = SVC(C=params.C, kernel=params.kernel, max_iter=params.maxiter,
          cache_size=params.cache_size_mb, tol=params.tol,
          shrinking=params.shrinking, gamma=params.gamma)

# Time fit and predict
fit_time, _ = bench.measure_function_time(clf.fit, X_train, y_train, params=params)
params.sv_len = clf.support_.shape[0]

predict_time, y_pred = bench.measure_function_time(
    clf.predict, X_train, params=params)
train_acc = 100 * accuracy_score(y_pred, y_train)

y_pred = clf.predict(X_test)
test_acc = 100 * accuracy_score(y_pred, y_test)

bench.print_output(library='sklearn', algorithm='svc',
                   stages=['training', 'prediction'],
                   params=params, functions=['SVM.fit', 'SVM.predict'],
                   times=[fit_time, predict_time], accuracy_type='accuracy[%]',
                   accuracies=[train_acc, test_acc], data=[X_train, X_train],
                   alg_instance=clf)
