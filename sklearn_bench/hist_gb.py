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
from sklearn.experimental import enable_hist_gradient_boosting  # noqa


def convert_probs_to_classes(y_prob):
    return np.array([np.argmax(y_prob[i]) for i in range(y_prob.shape[0])])


def convert_predictions(y_pred, objective, metric_name):
    if not objective.startswith('reg'):
        if metric_name == 'accuracy':
            y_pred = convert_probs_to_classes(y_pred)
    return y_pred


def main():
    # Load and convert data
    X_train, X_test, y_train, y_test = bench.load_data(params)

    if params.objective.startswith('reg'):
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=params.n_estimators,
                                              n_iter_no_change=params.n_estimators,
                                              learning_rate=params.learning_rate,
                                              max_bins=params.max_bin,
                                              max_depth=params.max_depth,
                                              max_leaf_nodes=params.max_leaves,
                                              l2_regularization=params.reg_lambda,
                                              random_state=params.seed)
        metric_funs = [bench.r2_score, bench.rmse_score]
        metric_name = ['r2_score', 'rmse']
        task = 'regression'

    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=params.n_estimators,
                                               n_iter_no_change=params.n_estimators,
                                               learning_rate=params.learning_rate,
                                               max_bins=params.max_bin,
                                               max_depth=params.max_depth,
                                               max_leaf_nodes=params.max_leaves,
                                               l2_regularization=params.reg_lambda,
                                               random_state=params.seed)

        metric_funs = [bench.accuracy_score,
                       bench.log_loss]
        metric_name = ['accuracy', 'log_loss']
        task = 'classification'

    metrics = [[None] * 2 for i in range(len(metric_funs))]

    fit_time, _ = bench.measure_function_time(
        model.fit, X_train, y_train, params=params)

    if not params.objective.startswith('reg'):
        y_pred = model.predict_proba(X_train)
    else:
        y_pred = model.predict(X_train)

    for i, func in enumerate(metric_funs):
        metrics[i][0] = func(y_train, convert_predictions(
            y_pred, params.objective, metric_name[i]))

    if not params.objective.startswith('reg'):
        predict_time, y_pred = bench.measure_function_time(
            model.predict_proba, X_test, params=params)

    else:
        predict_time, y_pred = bench.measure_function_time(
            model.predict, X_test, params=params)

    for i, func in enumerate(metric_funs):
        metrics[i][1] = metric_funs[i](y_test, convert_predictions(
            y_pred, params.objective, metric_name[i]))

    bench.print_output(
        library='sklearn',
        algorithm=f'hist_gradient_boosted_trees_{task}',
        stages=['training', 'prediction'],
        params=params,
        functions=['model.fit', 'model.predict_proba'],
        times=[fit_time, predict_time],
        metric_type=metric_name,
        metrics=metrics,
        data=[X_train, X_test],
        alg_instance=model,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scikit-learn histogram-based gradient '
                                                 'boosting benchmark')

    parser.add_argument('--n-estimators', type=int, default=100,
                        help='The number of gradient boosted trees')
    parser.add_argument('--objective', type=str, required=True,
                        choices=('reg:squarederror', 'binary:logistic',
                                 'multi:softmax', 'multi:softprob'))
    parser.add_argument('--reg-lambda', type=float, default=0,
                        help='L2 regularization term on weights')
    parser.add_argument('--learning-rate', '--eta', type=float, default=0.3,
                        help='Step size shrinkage used in update '
                        'to prevents overfitting')
    parser.add_argument('--max-bin', type=int, default=255,
                        help='Maximum number of discrete bins to '
                        'bucket continuous features')
    parser.add_argument('--max-depth', type=int, default=6,
                        help='Maximum depth of a tree')
    parser.add_argument('--max-leaves', type=int, default=31,
                        help='The maximum number of leaves for each tree')

    params = bench.parse_args(parser)
    bench.run_with_context(params, main)
