# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row,\
    convert_data, rmse_score, get_dtype
import numpy as np
from sklearn.linear_model import Ridge

parser = argparse.ArgumentParser(description='scikit-learn ridge regression '
                                             'benchmark')
parser.add_argument('--no-fit-intercept', dest='fit_intercept', default=True,
                    action='store_false',
                    help="Don't fit intercept (assume data already centered)")
parser.add_argument('--solver', default='auto',
                    help='Solver used for training')
params = parse_args(parser, size=(1000000, 50),
                    loop_types=('fit', 'predict'))

# Generate random data
X = convert_data(np.random.rand(*params.shape),
                 params.dtype, params.data_order, params.data_format)
Xp = convert_data(np.random.rand(*params.shape),
                  params.dtype, params.data_order, params.data_format)
y = convert_data(np.random.rand(*params.shape),
                 params.dtype, params.data_order, params.data_format)

# Create our regression object
regr = Ridge(fit_intercept=params.fit_intercept,
             solver=params.solver)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'time')
params.dtype = get_dtype(X)

# Time fit
fit_time, _ = time_mean_min(regr.fit, X, y,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)

# Time predict
predict_time, yp = time_mean_min(regr.predict, Xp,
                                 outer_loops=params.predict_outer_loops,
                                 inner_loops=params.predict_inner_loops,
                                 goal_outer_loops=params.predict_goal,
                                 time_limit=params.predict_time_limit,
                                 verbose=params.verbose)

rmse = rmse_score(y, yp)

if params.output_format == "csv":
    print_header(columns, params)
    print_row(columns, params, function='Ridge.fit', time=fit_time)
    print_row(columns, params, function='Ridge.predict', time=predict_time)
elif params.output_format == "json":
    import json

    res = {
        "lib": "sklearn",
        "algorithm": "ridge",
        "stage": "training",
        "data_format": params.data_format,
        "data_type": str(params.dtype),
        "data_order": params.data_order,
        "rows": X.shape[0],
        "columns": X.shape[1],
        "time[s]": fit_time,
        "algorithm_paramaters": dict(regr.get_params())
    }

    print(json.dumps(res, indent=4))

    res.update({
        "rows": Xp.shape[0],
        "columns": Xp.shape[1],
        "stage": "prediction",
        "time[s]": predict_time,
        "rmse": rmse
    })

    print(json.dumps(res, indent=4))
