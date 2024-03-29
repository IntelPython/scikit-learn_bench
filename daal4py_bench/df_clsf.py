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
from daal4py import (decision_forest_classification_prediction,
                     decision_forest_classification_training, engines_mt2203)
from daal4py.sklearn._utils import getFPType
from sklearn.metrics import accuracy_score


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

    parser.add_argument('--criterion', type=str, default='gini',
                        choices=('gini'),
                        help='The function to measure the quality of a split')
    parser.add_argument('--num-trees', type=int, default=100,
                        help='Number of trees in the forest')
    parser.add_argument('--max-features', type=bench.float_or_int, default=0,
                        help='Upper bound on features used at each split')
    parser.add_argument('--max-depth', type=int, default=0,
                        help='Upper bound on depth of constructed trees')
    parser.add_argument('--min-samples-split', type=bench.float_or_int, default=2,
                        help='Minimum samples number for node splitting')
    parser.add_argument('--max-leaf-nodes', type=int, default=None,
                        help='Maximum leaf nodes per tree')
    parser.add_argument('--min-impurity-decrease', type=float, default=0.,
                        help='Needed impurity decrease for node splitting')
    parser.add_argument('--no-bootstrap', dest='bootstrap', default=True,
                        action='store_false',
                        help="Don't control bootstraping")

    params = bench.parse_args(parser, prefix='daal4py')

    # Load data
    X_train, X_test, y_train, y_test = bench.load_data(
        params, add_dtype=True, label_2d=True)

    params.n_classes = len(np.unique(y_train))
    if isinstance(params.max_features, float):
        params.max_features = int(X_train.shape[1] * params.max_features)

    # Time fit and predict
    fit_time, res = bench.measure_function_time(
        df_clsf_fit, X_train, y_train,
        params.n_classes,
        n_trees=params.num_trees,
        n_features_per_node=params.max_features,
        max_depth=params.max_depth,
        min_impurity=params.min_impurity_decrease,
        bootstrap=params.bootstrap,
        seed=params.seed,
        params=params)

    yp = df_clsf_predict(X_train, res, params.n_classes)
    train_acc = 100 * accuracy_score(yp, y_train)

    predict_time, yp = bench.measure_function_time(
        df_clsf_predict, X_test, res, params.n_classes, params=params)
    test_acc = 100 * accuracy_score(yp, y_test)

    bench.print_output(library='daal4py', algorithm='decision_forest_classification',
                       stages=['training', 'prediction'], params=params,
                       functions=['df_clsf.fit', 'df_clsf.predict'],
                       times=[fit_time, predict_time], metric_type='accuracy[%]',
                       metrics=[train_acc, test_acc], data=[X_train, X_test])
