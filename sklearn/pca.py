# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from bench import parse_args, time_mean_min, print_header, print_row, convert_data
import numpy as np
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='scikit-learn PCA benchmark')
parser.add_argument('--svd-solver', type=str, choices=['daal', 'full'],
                    default='daal', help='SVD solver to use')
parser.add_argument('--n-components', type=int, default=None,
                    help='Number of components to find')
parser.add_argument('--whiten', action='store_true', default=False,
                    help='Perform whitening')
params = parse_args(parser, size=(10000, 1000),
                    loop_types=('fit', 'transform'))

# Generate random data
p, n = params.shape

X = convert_data(np.random.rand(*params.shape),
    params.dtype, params.data_order, params.data_format)
Xp = convert_data(np.random.rand(*params.shape),
    params.dtype, params.data_order, params.data_format)

if not params.n_components:
    params.n_components = min((n, (2 + min((n, p))) // 3))

# Create our PCA object
pca = PCA(svd_solver=params.svd_solver, whiten=params.whiten,
          n_components=params.n_components)

columns = ('batch', 'arch', 'prefix', 'function', 'threads', 'dtype', 'size',
           'svd_solver', 'n_components', 'whiten', 'time')

# Time fit
fit_time, _ = time_mean_min(pca.fit, X,
                            outer_loops=params.fit_outer_loops,
                            inner_loops=params.fit_inner_loops,
                            goal_outer_loops=params.fit_goal,
                            time_limit=params.fit_time_limit,
                            verbose=params.verbose)

# Time transform
transform_time, _ = time_mean_min(pca.transform, Xp,
                                  outer_loops=params.transform_outer_loops,
                                  inner_loops=params.transform_inner_loops,
                                  goal_outer_loops=params.transform_goal,
                                  time_limit=params.transform_time_limit,
                                  verbose=params.verbose)

if params.output_format == "csv":
    print_header(columns, params)
    print_row(columns, params, function='PCA.fit', time=fit_time)
    print_row(columns, params, function='PCA.transform', time=transform_time)
elif params.output_format == "json":
    import json

    res = {
        "lib": "sklearn",
        "algorithm": "pca",
        "stage": "training",
        "data_format": params.data_format,
        "data_type": str(params.dtype),
        "data_order": params.data_order,
        "rows": X.shape[0],
        "columns": X.shape[1],
        "n_components": params.n_components,
        "time[s]": fit_time,
        "algorithm_paramaters": dict(regr.get_params())
    }

    print(json.dumps(res, indent=4))

    res.update({
        "rows": Xp.shape[0],
        "columns": Xp.shape[1],
        "stage": "transform",
        "time[s]": transform_time
    })

    print(json.dumps(res, indent=4))
