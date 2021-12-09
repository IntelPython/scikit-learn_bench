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
    from sklearn.ensemble import RandomForestClassifier

    # Load and convert data
    X_train, X_test, y_train, y_test = bench.load_data(params)

    # Create our random forest classifier
    clf = RandomForestClassifier(criterion=params.criterion,
                                 n_estimators=params.num_trees,
                                 max_depth=params.max_depth,
                                 max_features=params.max_features,
                                 min_samples_split=params.min_samples_split,
                                 max_leaf_nodes=params.max_leaf_nodes,
                                 min_impurity_decrease=params.min_impurity_decrease,
                                 bootstrap=params.bootstrap,
                                 random_state=params.seed,
                                 n_jobs=params.n_jobs)

    params.n_classes = len(np.unique(y_train))

    fit_time, _ = bench.measure_function_time(clf.fit, X_train, y_train, params=params)
    y_pred = clf.predict(X_train)
    y_proba = clf.predict_proba(X_train)
    train_acc = bench.accuracy_score(y_train, y_pred)
    train_log_loss = bench.log_loss(y_train, y_proba)
    train_roc_auc = bench.roc_auc_score(y_train, y_proba)

    predict_time, y_pred = bench.measure_function_time(
        clf.predict, X_test, params=params)
    y_proba = clf.predict_proba(X_test)
    test_acc = bench.accuracy_score(y_test, y_pred)
    test_log_loss = bench.log_loss(y_test, y_proba)
    test_roc_auc = bench.roc_auc_score(y_test, y_proba)

    bench.print_output(
        library='sklearn',
        algorithm='df_clsf',
        stages=['training', 'prediction'],
        params=params,
        functions=['df_clsf.fit', 'df_clsf.predict'],
        times=[fit_time, predict_time],
        metric_type=['accuracy', 'log_loss', 'roc_auc'],
        metrics=[
            [train_acc, test_acc],
            [train_log_loss, test_log_loss],
            [train_roc_auc, test_roc_auc],
        ],
        data=[X_train, X_test],
        alg_instance=clf,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scikit-learn random forest '
                                                 'classification benchmark')

    parser.add_argument('--criterion', type=str, default='gini',
                        choices=('gini', 'entropy'),
                        help='The function to measure the quality of a split')
    parser.add_argument('--num-trees', type=int, default=100,
                        help='Number of trees in the forest')
    parser.add_argument('--max-features', type=bench.float_or_int_or_str, default=None,
                        help='Upper bound on features used at each split')
    parser.add_argument('--max-depth', type=int, default=None,
                        help='Upper bound on depth of constructed trees')
    parser.add_argument('--min-samples-split', type=bench.float_or_int, default=2,
                        help='Minimum samples number for node splitting')
    parser.add_argument('--max-leaf-nodes', type=int, default=None,
                        help='Maximum leaf nodes per tree')
    parser.add_argument('--min-impurity-decrease', type=float, default=0.,
                        help='Needed impurity decrease for node splitting')
    parser.add_argument('--no-bootstrap', dest='bootstrap', default=True,
                        action='store_false', help="Don't control bootstraping")

    params = bench.parse_args(parser)
    bench.run_with_context(params, main)
