# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT


import argparse
from bench import parse_args, time_mean_min, print_header, print_row, size_str
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
    parser.add_argument('-x', '--filex', '--fileX',
                        type=argparse.FileType('r'), required=True,
                        help='Input file with features, in NPY format')
    parser.add_argument('-y', '--filey', '--fileY',
                        type=argparse.FileType('r'), required=True,
                        help='Input file with targets, in NPY format')

    parser.add_argument('--num-trees', type=int, default=100,
                        help='Number of trees in the forest')
    parser.add_argument('--max-features', type=int, default=0,
                        help='Upper bound on features used at each split')
    parser.add_argument('--max-depth', type=int, default=0,
                        help='Upper bound on depth of constructed trees')

    parser.add_argument('--use-sklearn-class', action='store_true',
                        help='Force use of '
                             'sklearn.ensemble.RandomForestRegressor')
    parser.add_argument('--seed', type=int, default=12345,
                        help='Seed to pass as random_state to the class')
    params = parse_args(parser, loop_types=('fit', 'predict'),
                        prefix='daal4py')

    # Load data
    X = np.load(params.filex.name)
    y = np.load(params.filey.name)[:, np.newaxis]

    columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype',
               'size', 'num_trees', 'time')
    params.size = size_str(X.shape)
    params.dtype = X.dtype

    print_header(columns, params)

    # Time fit and predict
    fit_time, res = time_mean_min(df_regr_fit, X, y,
                                  n_trees=params.num_trees,
                                  seed=params.seed,
                                  n_features_per_node=params.max_features,
                                  max_depth=params.max_depth,
                                  outer_loops=params.fit_outer_loops,
                                  inner_loops=params.fit_inner_loops)
    print_row(columns, params, function='df_regr.fit', time=fit_time)

    predict_time, yp = time_mean_min(df_regr_predict, X, res,
                                     outer_loops=params.predict_outer_loops,
                                     inner_loops=params.predict_inner_loops)
    print_row(columns, params, function='df_regr.predict', time=predict_time)
