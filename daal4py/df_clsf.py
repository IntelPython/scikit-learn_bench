# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row, \
                  size_str, accuracy_score
import numpy as np
from daal4py import decision_forest_classification_training, \
                    decision_forest_classification_prediction, \
                    engines_mt2203
from daal4py.sklearn.utils import getFPType


def df_clsf_fit(X, y, n_classes, n_trees=100, seed=12345,
                n_features_per_node=0, max_depth=0, min_impurity=0,
                bootstrap=True, verbose=False):

    fptype = getFPType(X)

    features_per_node = X.shape[1]
    if n_features_per_node > 0 and n_features_per_node < features_per_node:
        features_per_node = n_features_per_node

    engine = engines_mt2203(seed=seed, fptype=fptype)

    algorithm = decision_forest_classification_training(
            nClasses=n_classes,
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

    df_clsf_result = algorithm.compute(X, y)

    return df_clsf_result


def df_clsf_predict(X, training_result, n_classes, verbose=False):

    algorithm = decision_forest_classification_prediction(
            nClasses=n_classes,
            fptype='float',  # we give float here specifically to match sklearn
    )

    result = algorithm.compute(X, training_result.model)

    return result.prediction


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='daal4py random forest '
                                                 'classification benchmark')

    parser.add_argument('-x', '--filex', '--fileX',
                        type=argparse.FileType('r'), required=True,
                        help='Input file with features, in NPY format')
    parser.add_argument('-y', '--filey', '--fileY',
                        type=argparse.FileType('r'), required=True,
                        help='Input file with labels, in NPY format')

    parser.add_argument('--num-trees', type=int, default=100,
                        help='Number of trees in the forest')
    parser.add_argument('--max-features',   type=int, default=0,
                        help='Upper bound on features used at each split')
    parser.add_argument('--max-depth',   type=int, default=0,
                        help='Upper bound on depth of constructed trees')

    parser.add_argument('--use-sklearn-class', action='store_true',
                        help='Force use of '
                             'sklearn.ensemble.RandomForestClassifier')
    parser.add_argument('--seed', type=int, default=12345,
                        help='Seed to pass as random_state to the class')
    params = parse_args(parser, loop_types=('fit', 'predict'),
                        prefix='daal4py')

    # Load data
    X = np.load(params.filex.name)
    y = np.load(params.filey.name)[:, np.newaxis]

    columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype',
               'size', 'num_trees', 'n_classes', 'accuracy', 'time')
    params.n_classes = len(np.unique(y))
    params.size = size_str(X.shape)
    params.dtype = X.dtype

    print_header(columns, params)

    # Time fit and predict
    fit_time, res = time_mean_min(df_clsf_fit, X, y, params.n_classes,
                                  n_trees=params.num_trees,
                                  seed=params.seed,
                                  n_features_per_node=params.max_features,
                                  max_depth=params.max_depth,
                                  verbose=params.verbose,
                                  outer_loops=params.fit_outer_loops,
                                  inner_loops=params.fit_inner_loops)
    print_row(columns, params, function='df_clsf.fit', time=fit_time)

    predict_time, yp = time_mean_min(df_clsf_predict, X, res,
                                     params.n_classes,
                                     verbose=params.verbose,
                                     outer_loops=params.predict_outer_loops,
                                     inner_loops=params.predict_inner_loops)
    acc = 100 * accuracy_score(yp, y)
    print_row(columns, params, function='df_clsf.predict', time=predict_time,
              accuracy=acc)
