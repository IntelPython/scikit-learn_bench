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
import catboost as cb
import daal4py
import typing as tp


def convert_probs_to_classes(y_prob, class_labels):
    return np.array([class_labels[np.argmax(y_prob[i])]
                    for i in range(y_prob.shape[0])])


def convert_cb_predictions(y_pred, objective, metric_name, class_labels):
    if objective != 'RMSE':
        if metric_name == 'accuracy':
            y_pred = convert_probs_to_classes(y_pred, class_labels)
    return y_pred


parser = argparse.ArgumentParser(
    description='catboost gbt + model transform + daal predict benchmark')

parser.add_argument('--count-pool', default=False, action='store_true',
                    help='Count Pool creation in time measurements')

parser.add_argument('--grow-policy', type=str, default='Depthwise',
                    help='Controls a way new nodes are added to the tree')

parser.add_argument('--learning-rate', '--eta', type=float, default=0.3,
                    help='Step size shrinkage used in update '
                    'to prevents overfitting')

parser.add_argument('--max-bin', type=int, default=256,
                    help='Maximum number of discrete bins to '
                    'bucket continuous features')

parser.add_argument('--max-depth', type=int, default=6,
                    help='Maximum depth of a tree')

parser.add_argument('--max-leaves', type=int, default=0,
                    help='Maximum number of nodes to be added')

parser.add_argument('--n-estimators', type=int, default=100,
                    help='Number of gradient boosted trees')

parser.add_argument('--objective', type=str, required=True,
                    choices=('RMSE', 'Logloss',
                             'multi:softmax', 'multi:softprob'),
                    help='Control a balance of positive and negative weights')

parser.add_argument('--reg-lambda', type=float, default=1,
                    help='L2 regularization term on weights')

parser.add_argument('--scale-pos-weight', type=float, default=1,
                    help='Controls a balance of positive and negative weights')

parser.add_argument('--subsample', type=float, default=1,
                    help='Subsample ratio of the training instances')

params = bench.parse_args(parser)

X_train, X_test, y_train, y_test = bench.load_data(params)

cb_params = {
    'verbose': 0,
    'learning_rate': params.learning_rate,
    'max_depth': params.max_depth,
    'subsample': params.subsample,
    'colsample_bylevel': 1,
    'reg_lambda': params.reg_lambda,
    'grow_policy': params.grow_policy,
    'max_bin': params.max_bin,
    'objective': params.objective,
    'random_seed': params.seed,
    'iterations': params.n_estimators,
}

# CatBoost restriction
if cb_params['grow_policy'] == 'Lossguide':
    cb_params['max_leaves'] = params.max_leaves

if params.threads != -1:
    cb_params.update({'thread_count': params.threads})

metric_name: tp.List[str]
metric_func: tp.List[tp.Callable]

class_labels = None

if params.objective == "RMSE":
    task = 'regression'
    metric_name = ['rmse', 'r2']
    metric_func = [bench.rmse_score, bench.r2_score]
else:
    task = 'classification'
    class_labels = sorted(np.unique(y_train))
    if params.objective.startswith('multi'):
        metric_name = ['accuracy']
        metric_func = [bench.accuracy_score]
    else:
        metric_name = ['accuracy', 'log_loss']
        metric_func = [bench.accuracy_score, bench.log_loss]

    if 'cudf' in str(type(y_train)):
        params.n_classes = y_train[y_train.columns[0]].nunique()
    else:
        params.n_classes = len(np.unique(y_train))
        unique_y_train = np.unique(y_train)
        params.n_classes = len(unique_y_train)

    if params.n_classes > 2:
        cb_params['bootstrap_type'] = 'Bernoulli'
        cb_params['objective'] = 'MultiClass'
    else:
        cb_params['scale_pos_weight'] = params.scale_pos_weight
        cb_params['objective'] = 'Logloss'

t_create_train, dtrain = bench.measure_function_time(
    cb.Pool, X_train, params=params, label=y_train)

t_create_test, dtest = bench.measure_function_time(
    cb.Pool, X_test, params=params, label=y_test)


def fit(pool):
    if pool is None:
        pool = cb.Pool(X_train, label=y_train)
    return cb.CatBoost(cb_params).fit(pool)


if params.objective.startswith('multi'):
    def predict(pool):
        if pool is None:
            pool = cb.Pool(X_test, label=y_test)
        return booster.predict(pool, prediction_type='Probability')
else:
    if cb_params['objective'] == 'Logloss':
        def predict(pool):
            if pool is None:
                pool = cb.Pool(X_test, label=y_test)
            return booster.predict(pool, prediction_type='Probability')
    else:
        def predict(pool):
            if pool is None:
                pool = cb.Pool(X_test, label=y_test)
            return booster.predict(pool)


fit_time, booster = bench.measure_function_time(
    fit, None if params.count_pool else dtrain, params=params)

# Create array where each metric has all the stages
metrics = [[None] * 6 for i in range(len(metric_name))]

# Metrics for training
for i, func in enumerate(metric_func):
    metrics[i][1] = func(
        y_train,
        convert_cb_predictions(
            predict(dtrain),
            params.objective,
            metric_name[i],
            class_labels))

predict_time, y_pred = bench.measure_function_time(
    predict, None if params.count_pool else dtest, params=params)

# Metrics for _prediction
for i, func in enumerate(metric_func):
    metrics[i][3] = func(y_test, convert_cb_predictions(
        y_pred, params.objective, metric_name[i], class_labels))

transform_time, model_daal = bench.measure_function_time(
    daal4py.get_gbt_model_from_catboost, booster, params=params)

if hasattr(params, 'n_classes'):
    predict_algo = daal4py.gbt_classification_prediction(
        nClasses=params.n_classes,
        resultsToEvaluate='computeClassProbabilities',
        fptype='float')
    predict_time_daal, daal_pred = bench.measure_function_time(
        predict_algo.compute, X_test, model_daal, params=params)
    daal_pred_value = daal_pred.probabilities
else:
    predict_algo = daal4py.gbt_regression_prediction()
    predict_time_daal, daal_pred = bench.measure_function_time(
        predict_algo.compute, X_test, model_daal, params=params)
    daal_pred_value = daal_pred.prediction

# Metrics for alternative_prediction
for i, func in enumerate(metric_func):
    metrics[i][5] = func(y_test, convert_cb_predictions(
        daal_pred_value, params.objective, metric_name[i], class_labels))

bench.print_output(
    library='modelbuilders',
    algorithm=f'catboost_{task}_and_modelbuilder',
    stages=[
        'training_preparation',
        'training',
        'prediction_preparation',
        'prediction',
        'transformation',
        'alternative_prediction'],
    params=params,
    functions=[
        'cb.Pool.train',
        'cb.fit',
        'cb.Pool.test',
        'cb.predict',
        'daal4py.get_gbt_model_from_catboost',
        'daal4py.compute'],
    times=[
        t_create_train,
        fit_time,
        t_create_test,
        predict_time,
        transform_time,
        predict_time_daal],
    metric_type=metric_name,
    metrics=metrics,
    data=[
        X_train,
        X_train,
        X_test,
        X_test,
        X_test,
        X_test])
