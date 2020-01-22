# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row,\
    size_str, convert_data, rmse_score
import numpy as np

parser = argparse.ArgumentParser(description='scikit-learn random forest '
                                             'regression benchmark')
parser.add_argument('-x', '--filex', '--fileX', type=argparse.FileType('r'),
                    required=True,
                    help='Input file with features, in NPY format')
parser.add_argument('-y', '--filey', '--fileY', type=argparse.FileType('r'),
                    required=True,
                    help='Input file with targets, in NPY format')

parser.add_argument('--num-trees', type=int, default=100,
                    help='Number of trees in the forest')
parser.add_argument('--max-features', type=int, default=None,
                    help='Upper bound on features used at each split')
parser.add_argument('--max-depth', type=int, default=None,
                    help='Upper bound on depth of constructed trees')

parser.add_argument('--use-sklearn-class', action='store_true',
                    help='Force use of '
                         'sklearn.ensemble.RandomForestRegressor')
parser.add_argument('--seed', type=int, default=12345,
                    help='Seed to pass as random_state to the class')
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
X = convert_data(np.load(params.filex.name),
    params.dtype, params.data_order, params.data_format)
y = convert_data(np.load(params.filey.name),
    params.dtype, params.data_order, params.data_format)

# Create our random forest regressor
regr = RandomForestRegressor(n_estimators=params.num_trees,
                             max_depth=params.max_depth,
                             max_features=params.max_features,
                             random_state=params.seed)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'num_trees', 'time')
params.size = size_str(X.shape)

fit_time, _ = time_mean_min(regr.fit, X, y,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)

predict_time, yp = time_mean_min(regr.predict, X,
                                     outer_loops=params.predict_outer_loops,
                                     inner_loops=params.predict_inner_loops,
                                     goal_outer_loops=params.predict_goal,
                                     time_limit=params.predict_time_limit,
                                     verbose=params.verbose)

rmse = rmse_score(yp, y)

if params.output_format == "csv":
    print_header(columns, params)
    print_row(columns, params, function='df_regr.fit', time=fit_time)
    print_row(columns, params, function='df_regr.predict', time=predict_time)
elif params.output_format == "json":
    import json

    res = {
        "lib": "sklearn",
        "algorithm": "decision_forest_regression",
        "stage": "training",
        "data_format": params.data_format,
        "data_type": str(params.dtype),
        "data_order": params.data_order,
        "rows": X.shape[0],
        "columns": X.shape[1],
        "time[s]": fit_time,
        "rmse": rmse,
        "algorithm_paramaters": dict(regr.get_params())
    }

    print(json.dumps(res, indent=4))

    res.update({
        "stage": "prediction",
        "time[s]": predict_time
    })

    print(json.dumps(res, indent=4))
