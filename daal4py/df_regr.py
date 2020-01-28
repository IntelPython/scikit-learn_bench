# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT


import argparse
from bench import (
    parse_args, time_mean_min, output_csv, load_data, gen_basic_dict,
    rmse_score
)
from daal4py import decision_forest_regression_training, \
                    decision_forest_regression_prediction, \
                    engines_mt2203
from daal4py.sklearn.utils import getFPType
import numpy as np


def df_regr_fit(X, y, n_trees=100, seed=12345, n_features_per_node=0,
                max_depth=0, min_impurity=0, bootstrap=True):

    fptype = getFPType(X)

    features_per_node = X.shape[1]
    if n_features_per_node > 0 and n_features_per_node <= features_per_node:
        features_per_node = n_features_per_node

    engine = engines_mt2203(seed=seed, fptype=fptype)

    algorithm = decision_forest_regression_training(
            fptype=fptype,
            method='defaultDense',
            nTrees=n_trees,
            observationsPerTreeFraction=1.,
            featuresPerNode=features_per_node,
            maxTreeDepth=max_depth,
            minObservationsInLeafNode=1,
            engine=engine,
            impurityThreshold=min_impurity,
            varImportance='MDI',
            resultsToCompute='',
            memorySavingMode=False,
            bootstrap=bootstrap
    )

    df_regr_result = algorithm.compute(X, y)

    return df_regr_result


def df_regr_predict(X, training_result):

    algorithm = decision_forest_regression_prediction(
            fptype='float'
    )

    result = algorithm.compute(X, training_result.model)

    return result.prediction


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='daal4py random forest '
                                                 'regression benchmark')

    parser.add_argument('--num-trees', type=int, default=100,
                        help='Number of trees in the forest')
    parser.add_argument('--max-features', type=int, default=0,
                        help='Upper bound on features used at each split')
    parser.add_argument('--max-depth', type=int, default=0,
                        help='Upper bound on depth of constructed trees')

    parser.add_argument('--use-sklearn-class', action='store_true',
                        help='Force use of '
                             'sklearn.ensemble.RandomForestRegressor')
    params = parse_args(parser, loop_types=('fit', 'predict'),
                        prefix='daal4py')

    # Load data
    X_train, X_test, y_train, y_test = load_data(
        params, add_dtype=True, label_2d=True)

    columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype',
               'size', 'num_trees', 'time')


    # Time fit and predict
    fit_time, res = time_mean_min(df_regr_fit, X_train, y_train,
                                  n_trees=params.num_trees,
                                  seed=params.seed,
                                  n_features_per_node=params.max_features,
                                  max_depth=params.max_depth,
                                  outer_loops=params.fit_outer_loops,
                                  inner_loops=params.fit_inner_loops,
                                  goal_outer_loops=params.fit_goal,
                                  time_limit=params.fit_time_limit,
                                  verbose=params.verbose)

    yp = df_regr_predict(X_train, res)
    train_rmse = rmse_score(yp, y_train)

    predict_time, yp = time_mean_min(df_regr_predict, X_test, res,
                                     outer_loops=params.predict_outer_loops,
                                     inner_loops=params.predict_inner_loops,
                                     goal_outer_loops=params.predict_goal,
                                     time_limit=params.predict_time_limit,
                                     verbose=params.verbose)

    test_rmse = rmse_score(yp, y_test)

    if params.output_format == "csv":
        output_csv(columns, params, functions=['df_regr.fit', 'df_regr.predict'],
                   times=[fit_time, predict_time])
    elif params.output_format == "json":
        import json

        result = gen_basic_dict("daal4py", "decision_forest_regression",
                                "training", params, X_train)
        result.update({
            "time[s]": fit_time,
            "rmse": train_rmse
        })
        print(json.dumps(result, indent=4))

        result = gen_basic_dict("daal4py", "decision_forest_regression",
                                "prediction", params, X_test)
        result.update({
            "time[s]": predict_time,
            "rmse": test_rmse
        })
        print(json.dumps(result, indent=4))
