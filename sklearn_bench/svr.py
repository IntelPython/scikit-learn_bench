# ===============================================================================
# Copyright 2021 Intel Corporation
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
    from sklearn.svm import SVR

    X_train, X_test, y_train, y_test = bench.load_data(params)
    y_train = np.asfortranarray(y_train).ravel()

    if params.gamma is None:
        params.gamma = 1.0 / X_train.shape[1]

    cache_size_bytes = bench.get_optimal_cache_size(X_train.shape[0],
                                                    max_cache=params.max_cache_size)
    params.cache_size_mb = cache_size_bytes / 1024**2
    params.n_classes = len(np.unique(y_train))

    regr = SVR(C=params.C, epsilon=params.epsilon, kernel=params.kernel,
               cache_size=params.cache_size_mb, tol=params.tol, gamma=params.gamma,
               degree=params.degree)

    fit_time, _ = bench.measure_function_time(regr.fit, X_train, y_train, params=params)
    params.sv_len = regr.support_.shape[0]

    predict_train_time, y_pred = bench.measure_function_time(
        regr.predict, X_train, params=params)
    train_rmse = bench.rmse_score(y_train, y_pred)

    _, y_pred = bench.measure_function_time(
        regr.predict, X_test, params=params)
    test_rmse = bench.rmse_score(y_test, y_pred)

    bench.print_output(library='sklearn', algorithm='svr',
                       stages=['training', 'prediction'],
                       params=params, functions=['SVR.fit', 'SVR.predict'],
                       times=[fit_time, predict_train_time], accuracy_type='rmse',
                       accuracies=[train_rmse, test_rmse], data=[X_train, X_train],
                       alg_instance=regr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scikit-learn SVR benchmark')

    parser.add_argument('-C', dest='C', type=float, default=1.,
                        help='SVR regularization parameter')
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=.1,
                        help='Epsilon in the epsilon-SVR model')
    parser.add_argument('--kernel', choices=('linear', 'rbf', 'poly', 'sigmoid'),
                        default='linear', help='SVR kernel function')
    parser.add_argument('--degree', type=int, default=3,
                        help='Degree of the polynomial kernel function')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Parameter for kernel="rbf"')
    parser.add_argument('--max-cache-size', type=int, default=8,
                        help='Maximum cache size, in gigabytes, for SVR.')
    parser.add_argument('--tol', type=float, default=1e-3,
                        help='Tolerance passed to sklearn.svm.SVR')

    params = bench.parse_args(parser, loop_types=('fit', 'predict'))
    bench.run_with_context(params, main)
