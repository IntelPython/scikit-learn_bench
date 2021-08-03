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
import numpy as np


def main():
    from sklearn.svm import SVC

    X_train, X_test, y_train, y_test = bench.load_data(params)
    y_train = np.asfortranarray(y_train).ravel()

    if params.gamma is None:
        params.gamma = 1.0 / X_train.shape[1]

    cache_size_bytes = bench.get_optimal_cache_size(X_train.shape[0],
                                                    max_cache=params.max_cache_size)
    params.cache_size_mb = cache_size_bytes / 1024**2
    params.n_classes = len(np.unique(y_train))

    clf = SVC(C=params.C, kernel=params.kernel, cache_size=params.cache_size_mb,
              tol=params.tol, gamma=params.gamma, probability=params.probability,
              random_state=43, degree=params.degree)

    fit_time, _ = bench.measure_function_time(clf.fit, X_train, y_train, params=params)
    params.sv_len = clf.support_.shape[0]

    if params.probability:
        state_predict = 'predict_proba'
        clf_predict = clf.predict_proba
        y_proba_train = clf_predict(X_train)
        y_proba_test = clf_predict(X_test)
        train_log_loss = bench.log_loss(y_train, y_proba_train)
        test_log_loss = bench.log_loss(y_test, y_proba_test)
        train_roc_auc = bench.roc_auc_score(y_train, y_proba_train)
        test_roc_auc = bench.roc_auc_score(y_test, y_proba_test)
    else:
        state_predict = 'prediction'
        clf_predict = clf.predict
        train_log_loss = None
        test_log_loss = None
        train_roc_auc = None
        test_roc_auc = None

    predict_train_time, y_pred = bench.measure_function_time(
        clf_predict, X_train, params=params)
    train_acc = bench.accuracy_score(y_train, y_pred)

    _, y_pred = bench.measure_function_time(
        clf_predict, X_test, params=params)
    test_acc = bench.accuracy_score(y_test, y_pred)

    bench.print_output(
        library='sklearn',
        algorithm='svc',
        stages=['training', state_predict],
        params=params,
        functions=['SVM.fit', f'SVM.{state_predict}'],
        times=[fit_time, predict_train_time],
        metric_type=['accuracy', 'log_loss', 'roc_auc', 'n_sv'],
        metrics=[
            [train_acc, test_acc],
            [train_log_loss, test_log_loss],
            [train_roc_auc, test_roc_auc],
            [int(clf.n_support_.sum()), int(clf.n_support_.sum())],
        ],
        data=[X_train, X_train],
        alg_instance=clf,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scikit-learn SVM benchmark')

    parser.add_argument('-C', dest='C', type=float, default=1.0,
                        help='SVM regularization parameter')
    parser.add_argument('--kernel', choices=('linear', 'rbf', 'poly', 'sigmoid'),
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

    params = bench.parse_args(parser, loop_types=('fit', 'predict'))
    bench.run_with_context(params, main)
