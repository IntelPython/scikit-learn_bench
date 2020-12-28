#===============================================================================
# Copyright 2020 Intel Corporation
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
#===============================================================================

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench
from sklearn.ensemble import RandomForestRegressor

parser = argparse.ArgumentParser(description='scikit-learn random forest '
                                             'regression benchmark')

parser.add_argument('--criterion', type=str, default='mse',
                    choices=('mse', 'mae'),
                    help='The function to measure the quality of a split')
parser.add_argument('--num-trees', type=int, default=100,
                    help='Number of trees in the forest')
parser.add_argument('--max-features', type=bench.float_or_int, default=None,
                    help='Upper bound on features used at each split')
parser.add_argument('--max-depth', type=int, default=None,
                    help='Upper bound on depth of constructed trees')
parser.add_argument('--min-samples-split', type=bench.float_or_int, default=2,
                    help='Minimum samples number for node splitting')
parser.add_argument('--max-leaf-nodes', type=int, default=None,
                    help='Grow trees with max_leaf_nodes in best-first fashion'
                         'if it is not None')
parser.add_argument('--min-impurity-decrease', type=float, default=0.,
                    help='Needed impurity decrease for node splitting')
parser.add_argument('--no-bootstrap', dest='bootstrap', default=True,
                    action='store_false', help="Don't control bootstraping")

params = bench.parse_args(parser)

# Load and convert data
X_train, X_test, y_train, y_test = bench.load_data(params)

# Create our random forest regressor
regr = RandomForestRegressor(criterion=params.criterion,
                             n_estimators=params.num_trees,
                             max_depth=params.max_depth,
                             max_features=params.max_features,
                             min_samples_split=params.min_samples_split,
                             max_leaf_nodes=params.max_leaf_nodes,
                             min_impurity_decrease=params.min_impurity_decrease,
                             bootstrap=params.bootstrap,
                             random_state=params.seed)

fit_time, _ = bench.measure_function_time(regr.fit, X_train, y_train, params=params)

y_pred = regr.predict(X_train)
train_rmse = bench.rmse_score(y_pred, y_train)

predict_time, y_pred = bench.measure_function_time(
    regr.predict, X_test, params=params)
test_rmse = bench.rmse_score(y_pred, y_test)

bench.print_output(library='sklearn', algorithm='decision_forest_regression',
                   stages=['training', 'prediction'], params=params,
                   functions=['df_regr.fit', 'df_regr.predict'],
                   times=[fit_time, predict_time], accuracy_type='rmse',
                   accuracies=[train_rmse, test_rmse], data=[X_train, X_test],
                   alg_instance=regr)
