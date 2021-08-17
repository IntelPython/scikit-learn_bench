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
    from sklearn.neighbors import KNeighborsRegressor

    # Load generated data
    X_train, X_test, y_train, y_test = bench.load_data(params)
    params.n_classes = len(np.unique(y_train))

    # Create a regression object
    knn_regr = KNeighborsRegressor(n_neighbors=params.n_neighbors,
                                   weights=params.weights,
                                   algorithm=params.method,
                                   metric=params.metric,
                                   n_jobs=params.n_jobs)

    # Measure time and accuracy on fitting
    train_time, _ = bench.measure_function_time(
        knn_regr.fit, X_train, y_train, params=params)
    if params.task == 'regression':
        y_pred = knn_regr.predict(X_train)
        train_rmse = bench.rmse_score(y_train, y_pred)
        train_r2 = bench.r2_score(y_train, y_pred)

    # Measure time and accuracy on prediction
    if params.task == 'regression':
        predict_time, yp = bench.measure_function_time(knn_regr.predict, X_test,
                                                       params=params)
        test_rmse = bench.rmse_score(y_test, yp)
        test_r2 = bench.r2_score(y_test, yp)
    else:
        predict_time, _ = bench.measure_function_time(knn_regr.kneighbors, X_test,
                                                      params=params)

    if params.task == 'regression':
        bench.print_output(
            library='sklearn',
            algorithm=knn_regr._fit_method + '_knn_regr',
            stages=['training', 'prediction'],
            params=params,
            functions=['knn_regr.fit', 'knn_regr.predict'],
            times=[train_time, predict_time],
            metric_type=['rmse', 'r2_score'],
            metrics=[[train_rmse, test_rmse], [train_r2, test_r2]],
            data=[X_train, X_test],
            alg_instance=knn_regr,
        )
    else:
        bench.print_output(
            library='sklearn',
            algorithm=knn_regr._fit_method + '_knn_search',
            stages=['training', 'search'],
            params=params,
            functions=['knn_regr.fit', 'knn_regr.kneighbors'],
            times=[train_time, predict_time],
            metric_type=None,
            metrics=[],
            data=[X_train, X_test],
            alg_instance=knn_regr,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='scikit-learn kNN classifier benchmark')

    parser.add_argument('--task', default='regression', type=str,
                        choices=('search', 'regression'),
                        help='The type of kNN task: search or regression')
    parser.add_argument('--n-neighbors', default=5, type=int,
                        help='The number of neighbors to use')
    parser.add_argument('--weights', type=str, default='uniform',
                        help='The weight function to be used in prediction')
    parser.add_argument('--method', type=str, default='brute',
                        choices=('brute', 'kd_tree', 'ball_tree', 'auto'),
                        help='The method to find the nearest neighbors')
    parser.add_argument('--metric', type=str, default='euclidean',
                        help='The metric to calculate distances')
    params = bench.parse_args(parser)
    bench.run_with_context(params, main)
