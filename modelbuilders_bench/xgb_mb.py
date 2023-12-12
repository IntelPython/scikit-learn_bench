# ==============================================================================
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
# ==============================================================================

import argparse

import bench
import daal4py
import numpy as np
import xgboost as xgb


def convert_probs_to_classes(y_prob):
    return np.array([np.argmax(y_prob[i]) for i in range(y_prob.shape[0])])


def convert_xgb_predictions(y_pred, objective):
    if objective == "multi:softprob":
        y_pred = convert_probs_to_classes(y_pred)
    elif objective == "binary:logistic":
        y_pred = (y_pred >= 0.5).astype(np.int32)
    return y_pred


def shap_accuracy(new, ref):
    # broadcast all values into single column and calculate RMSE
    return bench.rmse_score(
        new.reshape(
            -1,
        ),
        ref.reshape(
            -1,
        ),
    )


parser = argparse.ArgumentParser(
    description="xgboost gbt + model transform + daal predict benchmark"
)

parser.add_argument(
    "--colsample-bytree",
    type=float,
    default=1,
    help="Subsample ratio of columns " "when constructing each tree",
)
parser.add_argument(
    "--count-dmatrix",
    default=False,
    action="store_true",
    help="Count DMatrix creation in time measurements",
)
parser.add_argument(
    "--enable-experimental-json-serialization",
    default=True,
    choices=("True", "False"),
    help="Use JSON to store memory snapshots",
)
parser.add_argument(
    "--grow-policy",
    type=str,
    default="depthwise",
    help="Controls a way new nodes are added to the tree",
)
parser.add_argument(
    "--inplace-predict",
    default=False,
    action="store_true",
    help="Perform inplace_predict instead of default",
)
parser.add_argument(
    "--learning-rate",
    "--eta",
    type=float,
    default=0.3,
    help="Step size shrinkage used in update " "to prevents overfitting",
)
parser.add_argument(
    "--max-bin",
    type=int,
    default=256,
    help="Maximum number of discrete bins to " "bucket continuous features",
)
parser.add_argument(
    "--max-delta-step",
    type=float,
    default=0,
    help="Maximum delta step we allow each leaf output to be",
)
parser.add_argument("--max-depth", type=int, default=6, help="Maximum depth of a tree")
parser.add_argument(
    "--max-leaves", type=int, default=0, help="Maximum number of nodes to be added"
)
parser.add_argument(
    "--min-child-weight",
    type=float,
    default=1,
    help="Minimum sum of instance weight needed in a child",
)
parser.add_argument(
    "--min-split-loss",
    "--gamma",
    type=float,
    default=0,
    help="Minimum loss reduction required to make" " partition on a leaf node",
)
parser.add_argument(
    "--n-estimators", type=int, default=100, help="Number of gradient boosted trees"
)
parser.add_argument(
    "--objective",
    type=str,
    required=True,
    choices=("reg:squarederror", "binary:logistic", "multi:softmax", "multi:softprob"),
    help="Control a balance of positive and negative weights",
)
parser.add_argument(
    "--reg-alpha", type=float, default=0, help="L1 regularization term on weights"
)
parser.add_argument(
    "--reg-lambda", type=float, default=1, help="L2 regularization term on weights"
)
parser.add_argument(
    "--scale-pos-weight",
    type=float,
    default=1,
    help="Controls a balance of positive and negative weights",
)
parser.add_argument(
    "--single-precision-histogram",
    default=False,
    action="store_true",
    help="Build histograms instead of double precision",
)
parser.add_argument(
    "--subsample",
    type=float,
    default=1,
    help="Subsample ratio of the training instances",
)
parser.add_argument(
    "--tree-method",
    type=str,
    required=True,
    help="The tree construction algorithm used in XGBoost",
)

params = bench.parse_args(parser)


X_train, X_test, y_train, y_test = bench.load_data(params)

xgb_params = {
    "booster": "gbtree",
    "verbosity": 0,
    "learning_rate": params.learning_rate,
    "min_split_loss": params.min_split_loss,
    "max_depth": params.max_depth,
    "min_child_weight": params.min_child_weight,
    "max_delta_step": params.max_delta_step,
    "subsample": params.subsample,
    "sampling_method": "uniform",
    "colsample_bytree": params.colsample_bytree,
    "colsample_bylevel": 1,
    "colsample_bynode": 1,
    "reg_lambda": params.reg_lambda,
    "reg_alpha": params.reg_alpha,
    "tree_method": params.tree_method,
    "scale_pos_weight": params.scale_pos_weight,
    "grow_policy": params.grow_policy,
    "max_leaves": params.max_leaves,
    "max_bin": params.max_bin,
    "objective": params.objective,
    "seed": params.seed,
    "single_precision_histogram": params.single_precision_histogram,
    "enable_experimental_json_serialization": params.enable_experimental_json_serialization,
}

xgb_params.update({"nthread": params.threads})
daal4py.daalinit(params.threads)

if params.objective.startswith("reg"):
    task = "regression"
    metric_name, metric_func = "rmse", bench.rmse_score
else:
    task = "classification"
    metric_name = "accuracy"
    metric_func = bench.accuracy_score
    if "cudf" in str(type(y_train)):
        params.n_classes = y_train[y_train.columns[0]].nunique()
    else:
        params.n_classes = len(np.unique(y_train))

    # Covtype has one class more than there is in train
    if params.dataset_name == "covtype":
        params.n_classes += 1

    if params.n_classes > 2:
        xgb_params["num_class"] = params.n_classes

t_creat_train, dtrain = bench.measure_function_time(
    xgb.DMatrix, X_train, params=params, label=y_train
)
t_creat_test, dtest = bench.measure_function_time(
    xgb.DMatrix, X_test, params=params, label=y_test
)

# SHAP interactions are very expensive - cap the number of rows
interaction_n_rows = max(2_000, 200_000 // (X_test.shape[0] * X_test.shape[1]))

# not benchmarked, but required for SHAP interactions
dtest_interactions = xgb.DMatrix(X_test[:interaction_n_rows])


def fit(dmatrix):
    if dmatrix is None:
        dmatrix = xgb.DMatrix(X_train, y_train)
    return xgb.train(xgb_params, dmatrix, params.n_estimators)


if params.inplace_predict:

    def predict(*args):
        return booster.inplace_predict(
            np.ascontiguousarray(X_test.values, dtype=np.float32)
        )

else:

    def predict(dmatrix, **kwargs):  # type: ignore
        if dmatrix is None:
            dmatrix = xgb.DMatrix(X_test, y_test)
        return booster.predict(dmatrix, **kwargs)


fit_time, booster = bench.measure_function_time(
    fit, None if params.count_dmatrix else dtrain, params=params
)
train_metric = metric_func(
    convert_xgb_predictions(booster.predict(dtrain), params.objective), y_train
)

predict_time, y_pred = bench.measure_function_time(
    predict,
    None if params.inplace_predict or params.count_dmatrix else dtest,
    params=params,
)
test_metric = metric_func(convert_xgb_predictions(y_pred, params.objective), y_test)

shap_contrib_time, shap_contribs = bench.measure_function_time(
    predict, dtest, pred_contribs=True, params=params
)

shap_interaction_time, shap_interactions = bench.measure_function_time(
    predict, dtest_interactions, pred_interactions=True, params=params
)

transform_time, model_daal = bench.measure_function_time(
    daal4py.mb.convert_model, booster, params=params
)

predict_time_daal, daal_pred = bench.measure_function_time(
    model_daal.predict, X_test, params=params
)
test_metric_daal = metric_func(y_test, daal_pred)

if model_daal._is_regression:
    shap_contrib_time_daal, daal_contribs = bench.measure_function_time(
        model_daal.predict, X_test, pred_contribs=True, params=params
    )

    shap_interaction_time_daal, daal_interactions = bench.measure_function_time(
        model_daal.predict,
        X_test[:interaction_n_rows],
        pred_interactions=True,
        params=params,
    )

    contrib_accuracy = shap_accuracy(shap_contribs, daal_contribs)

    interaction_accuracy = shap_accuracy(shap_interactions, daal_interactions)

else:
    # classification currently does not support SHAP values
    (
        shap_contrib_time_daal,
        shap_interaction_time_daal,
        contrib_accuracy,
        interaction_accuracy,
    ) = [0] * 4

bench.print_output(
    library="modelbuilders",
    algorithm=f"xgboost_{task}_and_modelbuilder",
    alg_instance=booster,
    alg_params={
        "max-depth": getattr(params, "max_depth", None),
        "objective": getattr(params, "objective", None),
    },
    stages=[
        "training_preparation",
        "training",
        "prediction_preparation",
        "prediction",
        "transformation",
        "alternative_prediction",
        "shap_contrib_prediction",
        "alternative_shap_contrib_prediction",
        "shap_interaction_prediction",
        "alternative_shap_interaction_prediction",
    ],
    data=[X_train] * 2 + [X_test] * 2 + [X_train] + [X_test] * 5,
    params=params,
    functions=[
        "xgb.dmatrix.train",
        "xgb.train",
        "xgb.dmatrix.test",
        "xgb.predict",
        "daal4py.get_gbt_model_from_xgboost",
        "daal4py.predict",
        "xgb.predict(pred_contribs=True)",
        "daal4py.predict(pred_contribs=True)",
        "xgb.predict(pred_interactions=True)",
        "daal4py.predict(pred_interactions=True)",
    ],
    times=[
        t_creat_train,
        fit_time,
        t_creat_test,
        predict_time,
        transform_time,
        predict_time_daal,
        shap_contrib_time,
        shap_contrib_time_daal,
        shap_interaction_time,
        shap_interaction_time_daal,
    ],
    metric_type=[metric_name, "rmse"],
    metrics=[
        [
            None,
            train_metric,
            None,
            test_metric,
            None,
            test_metric_daal,
            None,
            None,
            None,
            None,
        ],
        [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            contrib_accuracy,
            None,
            interaction_accuracy,
        ],
    ],
)
