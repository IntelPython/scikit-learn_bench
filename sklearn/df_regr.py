# Copyright (C) 2018-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import (
    parse_args, time_mean_min, output_csv, load_data, gen_basic_dict,
    rmse_score
)

parser = argparse.ArgumentParser(description='scikit-learn random forest '
                                             'regression benchmark')

parser.add_argument('--num-trees', type=int, default=100,
                    help='Number of trees in the forest')
parser.add_argument('--max-features', type=int, default=None,
                    help='Upper bound on features used at each split')
parser.add_argument('--max-depth', type=int, default=None,
                    help='Upper bound on depth of constructed trees')

parser.add_argument('--use-sklearn-class', action='store_true',
                    help='Force use of '
                         'sklearn.ensemble.RandomForestRegressor')
params = parse_args(parser, loop_types=('fit', 'predict'))

# Get some RandomForestRegressor
if params.use_sklearn_class:
    from sklearn.ensemble import RandomForestRegressor
else:
    try:
        from daal4py.sklearn.ensemble import RandomForestRegressor
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor

# Load and convert data
X_train, X_test, y_train, y_test = load_data(params)

# Create our random forest regressor
regr = RandomForestRegressor(n_estimators=params.num_trees,
                             max_depth=params.max_depth,
                             max_features=params.max_features,
                             random_state=params.seed)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'num_trees', 'time')

fit_time, _ = time_mean_min(regr.fit, X_train, y_train,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)

y_pred = regr.predict(X_train)
train_rmse = rmse_score(y_pred, y_train)

predict_time, y_pred = time_mean_min(regr.predict, X_test,
                                     outer_loops=params.predict_outer_loops,
                                     inner_loops=params.predict_inner_loops,
                                     goal_outer_loops=params.predict_goal,
                                     time_limit=params.predict_time_limit,
                                     verbose=params.verbose)
test_rmse = rmse_score(y_pred, y_test)

if params.output_format == 'csv':
    output_csv(columns, params, functions=['df_regr.fit', 'df_regr.predict'],
               times=[fit_time, predict_time])
elif params.output_format == 'json':
    import json

    result = gen_basic_dict('sklearn', 'decision_forest_regression',
                            'training', params, X_train, regr)
    result.update({
        'time[s]': fit_time,
        'rmse': train_rmse
    })
    print(json.dumps(result, indent=4))

    result = gen_basic_dict('sklearn', 'decision_forest_regression',
                            'prediction', params, X_test, regr)
    result.update({
        'time[s]': predict_time,
        'rmse': test_rmse
    })
    print(json.dumps(result, indent=4))
