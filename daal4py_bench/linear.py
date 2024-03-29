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
from daal4py import linear_regression_prediction, linear_regression_training
from daal4py.sklearn._utils import getFPType


parser = argparse.ArgumentParser(description='daal4py linear regression '
                                             'benchmark')
parser.add_argument('--no-fit-intercept', dest='fit_intercept', default=True,
                    action='store_false',
                    help="Don't fit intercept (assume data already centered)")
parser.add_argument('--method', default='normEqDense',
                    choices=('normEqDense', 'qrDense'),
                    help='Training method used by DAAL. "normEqDense" selects'
                         'the normal equations method, while "qrDense" selects'
                         'the method based on QR decomposition.')

params = bench.parse_args(parser, prefix='daal4py')

# Generate random data
X_train, X_test, y_train, y_test = bench.load_data(
    params, generated_data=['X_train', 'y_train'], add_dtype=True,
    label_2d=True if params.file_X_train is not None else False)


# Create our regression objects
def test_fit(X, y):
    regr_train = linear_regression_training(fptype=getFPType(X),
                                            method=params.method,
                                            interceptFlag=params.fit_intercept)
    return regr_train.compute(X, y)


def test_predict(Xp, model):
    regr_predict = linear_regression_prediction(fptype=getFPType(Xp))
    return regr_predict.compute(Xp, model)


# Time fit
fit_time, res = bench.measure_function_time(
    test_fit, X_train, y_train, params=params)

# Time predict
predict_time, pres = bench.measure_function_time(
    test_predict, X_test, res.model, params=params)

test_rmse = bench.rmse_score(pres.prediction, y_test)
pres = test_predict(X_train, res.model)
train_rmse = bench.rmse_score(pres.prediction, y_train)

bench.print_output(library='daal4py', algorithm='linear_regression',
                   stages=['training', 'prediction'],
                   params=params, functions=['Linear.fit', 'Linear.predict'],
                   times=[fit_time, predict_time], metric_type='rmse',
                   metrics=[train_rmse, test_rmse], data=[X_train, X_test])
