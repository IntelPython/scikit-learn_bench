# ===============================================================================
# Copyright 2020-2021 Intel Corporation
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
# ===============================================================================

import argparse

import bench
from cuml.svm import SVC


parser = argparse.ArgumentParser(description='cuML SVM benchmark')

parser.add_argument('-C', dest='C', type=float, default=1.0,
                    help='SVM regularization parameter')
parser.add_argument('--kernel', choices=('linear', 'rbf', 'poly'),
                    default='linear', help='SVM kernel function')
parser.add_argument('--degree', type=int, default=3,
                    help='Degree of the polynomial kernel function')
parser.add_argument('--gamma', type=float, default=None,
                    help='Parameter for kernel="rbf"')
parser.add_argument('--max-cache-size', type=int, default=8,
                    help='Maximum cache size, in gigabytes, for SVM.')
parser.add_argument('--tol', type=float, default=1e-3,
                    help='Tolerance passed to sklearn.svm.SVC')
parser.add_argument('--probability', action='store_true', default=False,
                    dest='probability', help="Use probability for SVC")

params = bench.parse_args(parser)

X_train, X_test, y_train, y_test = bench.load_data(params)

if params.gamma is None:
    params.gamma = 1.0 / X_train.shape[1]

cache_size_bytes = bench.get_optimal_cache_size(X_train.shape[0],
                                                max_cache=params.max_cache_size)
params.cache_size_mb = cache_size_bytes / 1024**2
params.n_classes = y_train[y_train.columns[0]].nunique()

clf = SVC(C=params.C, kernel=params.kernel, cache_size=params.cache_size_mb,
          tol=params.tol, gamma=params.gamma, probability=params.probability,
          degree=params.degree)

fit_time, _ = bench.measure_function_time(clf.fit, X_train, y_train, params=params)

if params.probability:
    state_predict = 'predict_proba'
    accuracy_type = 'log_loss'
    clf_predict = clf.predict_proba

    def metric_call(x, y):
        return bench.log_loss(x, y)
else:
    state_predict = 'prediction'
    accuracy_type = 'accuracy[%]'
    clf_predict = clf.predict

    def metric_call(x, y):
        return 100 * bench.accuracy_score(x, y)


predict_train_time, y_pred = bench.measure_function_time(
    clf_predict, X_train, params=params)
train_acc = metric_call(y_train, y_pred)

predict_test_time, y_pred = bench.measure_function_time(
    clf_predict, X_test, params=params)
test_acc = metric_call(y_test, y_pred)

bench.print_output(library='cuml', algorithm='svc',
                   stages=['training', state_predict], params=params,
                   functions=['SVM.fit', 'SVM.predict'],
                   times=[fit_time, predict_train_time], accuracy_type=accuracy_type,
                   accuracies=[train_acc, test_acc], data=[X_train, X_train],
                   alg_instance=clf)
