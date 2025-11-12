# ===============================================================================
# Copyright 2024 Intel Corporation
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
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.datasets import (
    fetch_california_housing,
    fetch_covtype,
    load_digits,
    load_svmlight_file,
    make_blobs,
    make_circles,
    make_classification,
    make_moons,
    make_regression,
)
from sklearn.datasets._base import fetch_file
from sklearn.preprocessing import StandardScaler

from .common import cache, load_data_description, load_data_from_cache, preprocess
from .downloaders import download_and_read_csv, load_openml, retrieve


@preprocess
@cache
def load_openml_data(
    openml_id: int, data_name: str, data_cache: str, raw_data_cache: str
) -> Tuple[Dict, Dict]:
    x, y = load_openml(openml_id, raw_data_cache)
    data_desc = dict()
    unique_labels = dict(pd.value_counts(y))
    if len(unique_labels) < 32 and all(map(lambda x: x > 4, unique_labels.values())):
        data_desc["n_classes"] = len(unique_labels)
    return {"x": x, "y": y}, data_desc


@preprocess
@cache
def load_sklearn_synthetic_data(
    function_name: str,
    input_kwargs: Dict,
    data_name: str,
    data_cache: str,
    raw_data_cache: str,
) -> Tuple[Dict, Dict]:
    functions_map = {
        "make_classification": make_classification,
        "make_regression": make_regression,
        "make_blobs": make_blobs,
        "make_moons": make_moons,
        "make_circles": make_circles,
    }
    generation_kwargs = {"random_state": 42}
    generation_kwargs.update(input_kwargs)

    if function_name not in functions_map:
        raise ValueError(
            f"Unknown {function_name} function " "for synthetic data generation"
        )
    x, y = functions_map[function_name](**generation_kwargs)
    data_desc = dict()
    if function_name == "make_classification":
        data_desc["n_classes"] = generation_kwargs["n_classes"]
        data_desc["n_clusters_per_class"] = generation_kwargs.get(
            "n_clusters_per_class", 2
        )
    elif function_name == "make_blobs":
        data_desc["n_clusters"] = generation_kwargs["centers"]
    elif function_name in ["make_circles", "make_moons"]:
        data_desc["n_classes"] = 2
        data_desc["n_clusters"] = 2
    return {"x": x, "y": y}, data_desc


@preprocess
def load_custom_data(
    data_name: str,
    data_cache: str,
    raw_data_cache: str,
):
    """Function to load data specified by user and stored in format compatible with scikit-learn_bench cache"""
    return load_data_from_cache(data_cache, data_name), load_data_description(
        data_cache, data_name
    )


"""
Classification datasets
"""


@cache
def load_airline_depdelay(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    Airline dataset
    http://kt.ijs.si/elena_ikonomovska/data.html

    Classification task. n_classes = 2.
    """

    ARCHIVE = (
        "airline_14col.data.bz2",
        "http://kt.ijs.si/elena_ikonomovska/datasets/airline",
        "1f13460fcdfb9b98f1b8932f2da3c23acc1ed3bdc906e5658c612be2849c74c5",
    )

    ordered_columns = [
        "Year",
        "Month",
        "DayofMonth",
        "DayofWeek",
        "CRSDepTime",
        "CRSArrTime",
        "UniqueCarrier",
        "FlightNum",
        "ActualElapsedTime",
        "Origin",
        "Dest",
        "Distance",
        "Diverted",
        "ArrDelay",
    ]
    categorical_int_columns = ["Year", "Month", "DayofMonth", "DayofWeek"]
    continuous_int_columns = [
        "CRSDepTime",
        "CRSArrTime",
        "FlightNum",
        "ActualElapsedTime",
        "Distance",
        "Diverted",
        "ArrDelay",
    ]
    column_dtypes = {
        col: np.int16 for col in categorical_int_columns + continuous_int_columns
    }

    df = download_and_read_csv(
        ARCHIVE, raw_data_cache, names=ordered_columns, dtype=column_dtypes
    )

    for col in df.select_dtypes(["object"]).columns:
        df[col] = df[col].astype("category")

    task = dataset_params.get("task", "classification")
    if task == "classification":
        df["ArrDelay"] = (df["ArrDelay"] > 0).astype(int)
    elif task == "regression":
        pass
    else:
        raise ValueError(f'Unknown "{task}" task type for airline dataset.')

    y = df["ArrDelay"].to_numpy(dtype=np.float32)
    x = df.drop(columns=["ArrDelay"])

    data_description = {
        "n_classes": 2,
        "default_split": {"test_size": 0.2, "random_state": 42},
    }
    return {"x": x, "y": y}, data_description


@cache
def load_hepmass(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    HEPMASS dataset from UCI machine learning repository
    https://archive.ics.uci.edu/ml/datasets/HEPMASS.

    Classification task. n_classes = 2.
    """

    BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00347"

    ARCHIVE_TRAIN = (
        "all_train.csv.gz",
        BASE_URL,
        "52061273edbe84cbfff6cc5432a04366d3401c39baf80da99d9baf91e0165498",
    )
    ARCHIVE_TEST = (
        "all_test.csv.gz",
        BASE_URL,
        "eccba00f8d82c471c582ab629084103356f8dda637fad6d43f16a056673091b3",
    )

    dtype = np.float32
    train_data = download_and_read_csv(
        ARCHIVE_TRAIN, raw_data_cache, delimiter=",", compression="gzip", dtype=dtype
    )
    test_data = download_and_read_csv(
        ARCHIVE_TEST, raw_data_cache, delimiter=",", compression="gzip", dtype=dtype
    )

    data = pd.concat([train_data, test_data])
    label = data.columns[0]
    y = data[label]
    x = data.drop(columns=[label, "mass"])

    data_desc = {
        "n_classes": 2,
        "default_split": {
            "train_size": train_data.shape[0],
            "test_size": test_data.shape[0],
            "shuffle": False,
        },
    }
    return {"x": x, "y": y}, data_desc


def load_higgs_susy_subsample(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    if data_name == "susy":
        """
        SUSY dataset from UCI machine learning repository
        https://archive.ics.uci.edu/ml/datasets/SUSY

        Classification task. n_classes = 2.
        """
        ARCHIVE = (
            "SUSY.csv.gz",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00279",
            "be56cb5598da8ece4b13912230ee713bab8b3431a7d118e0054ffdf3a2f25664",
        )

        train_size, test_size = 4500000, 500000
    elif data_name == "higgs":
        """
        Higgs dataset from UCI machine learning repository
        https://archive.ics.uci.edu/ml/datasets/HIGGS

        Classification task. n_classes = 2.
        """
        ARCHIVE = (
            "HIGGS.csv.gz",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00280",
            "ea302c18164d4e3d916a1e2e83a9a8d07069fa6ebc7771e4c0540d54e593b698",
        )

        train_size, test_size = 10000000, 1000000
    else:
        raise ValueError(
            f"Unknown dataset name {data_name} "
            'for "load_higgs_susy_subsample" function'
        )

    data = download_and_read_csv(
        ARCHIVE, raw_data_cache, delimiter=",", header=None, compression="gzip"
    )
    assert data.shape[0] == train_size + test_size, "Wrong number of samples was loaded"
    x, y = data[data.columns[1:]], data[data.columns[0]]

    data_desc = {
        "n_classes": 2,
        "default_split": {
            "train_size": train_size,
            "test_size": test_size,
            "shuffle": False,
        },
    }
    return {"x": x, "y": y}, data_desc


@cache
def load_higgs(**kwargs) -> Tuple[Dict, Dict]:
    return load_higgs_susy_subsample(**kwargs)


@cache
def load_susy(**kwargs) -> Tuple[Dict, Dict]:
    return load_higgs_susy_subsample(**kwargs)


@cache
def load_letters(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    Letter Recognition dataset from UCI machine learning repository
    http://archive.ics.uci.edu/ml/datasets/Letter+Recognition

    Classification task. n_classes = 26.
    """

    ARCHIVE = (
        "letter-recognition.data",
        "http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition",
        "2b89f3602cf768d3c8355267d2f13f2417809e101fc2b5ceee10db19a60de6e2",
    )

    data = download_and_read_csv(ARCHIVE, raw_data_cache, header=None, dtype=None)
    x, y = data.iloc[:, 1:], data.iloc[:, 0].astype("category").cat.codes.values

    data_desc = {"n_classes": 26, "default_split": {"test_size": 0.2, "random_state": 0}}
    return {"x": x, "y": y}, data_desc


@cache
def load_sklearn_digits(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    x, y = load_digits(return_X_y=True)
    data_desc = {
        "n_classes": 10,
        "default_split": {"train_size": 0.2, "random_state": 42},
    }
    return {"x": x, "y": y}, data_desc


@cache
def load_covtype(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    Cover type dataset from UCI machine learning repository
    https://archive.ics.uci.edu/ml/datasets/covertype

    y contains 7 unique class labels from 1 to 7 inclusive.
    Classification task. n_classes = 7.
    """
    x, y = fetch_covtype(return_X_y=True, data_home=raw_data_cache)
    y = y.astype(int) - 1
    binary = dataset_params.get("binary", False)
    if binary:
        y = (y > 2).astype(int)

    data_desc = {
        "n_classes": 2 if binary else 7,
        "default_split": {"test_size": 0.2, "random_state": 77},
    }
    return {"x": x, "y": y}, data_desc


@cache
def load_epsilon(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    Epsilon dataset
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

    Classification task. n_classes = 2.
    """
    ARCHIVE_TRAIN = (
        "epsilon_normalized.bz2",
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
        "aff916d4f97f18d286558ca088d2a9f7e1fcee9376539a5aa6ef5b7ef9dfa978",
    )

    ARCHIVE_TEST = (
        "epsilon_normalized.t.bz2",
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
        "cb299295ad11e200696eaa3050f5d8cf700eaa9c65e6aa859bda959f8669458b",
    )

    local_train_path = os.path.join(raw_data_cache, os.path.basename(ARCHIVE_TRAIN[0]))
    local_test_path = os.path.join(raw_data_cache, os.path.basename(ARCHIVE_TEST[0]))

    _ = fetch_file(
        url=ARCHIVE_TRAIN[1],
        folder=raw_data_cache,
        local_filename=ARCHIVE_TRAIN[0],
        sha256=ARCHIVE_TRAIN[2],
    )
    _ = fetch_file(
        url=ARCHIVE_TEST[1],
        folder=raw_data_cache,
        local_filename=ARCHIVE_TEST[0],
        sha256=ARCHIVE_TEST[2],
    )

    x_train, y_train = load_svmlight_file(local_train_path, dtype=np.float32)
    x_test, y_test = load_svmlight_file(local_test_path, dtype=np.float32)

    x = sparse.vstack([x_train, x_test])
    y = np.hstack([y_train, y_test])
    y[y <= 0] = 0

    data_desc = {
        "n_classes": 2,
        "default_split": {
            "train_size": y_train.shape[0],
            "test_size": y_test.shape[0],
            "shuffle": False,
        },
    }
    return {"x": x, "y": y}, data_desc


@cache
def load_gisette(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    GISETTE is a handwritten digit recognition problem.
    The problem is to separate the highly confusable digits '4' and '9'.
    This dataset is one of five datasets of the NIPS 2003 feature selection challenge.

    Classification task. n_classes = 2.
    """

    def convert_x(x, n_samples, n_features):
        x_out = x.iloc[:n_samples].values
        x_out = pd.DataFrame(
            np.array(
                [
                    np.fromstring(elem[0], dtype=int, count=n_features, sep=" ")
                    for elem in x_out
                ]
            )
        )
        return x_out.values

    def convert_y(y, n_samples):
        y_out = y.iloc[:n_samples].values.astype(int)
        y_out = pd.DataFrame((y_out > 0).astype(int))
        return y_out.values.reshape(-1)

    BASE_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases"

    data_meta = {
        "x_train": (
            "gisette_train.data",
            f"{BASE_URL}/gisette/GISETTE",
            "6d4c5e998afe67937b9e77a3334e03c85e545ebc65a6eb1333ffc14125cfc389",
        ),
        "x_test": (
            "gisette_valid.data",
            f"{BASE_URL}/gisette/GISETTE",
            "5cea897956dd172a006132738254a27a8f61ecc1ceb6f5b20639c281d2942254",
        ),
        "y_train": (
            "gisette_train.labels",
            f"{BASE_URL}/gisette/GISETTE",
            "42bd681fe51b161f033df773df14a0116e492676555ab14616c1b72edc054075",
        ),
        "y_test": (
            "gisette_valid.labels",
            f"{BASE_URL}/gisette",
            "a6b857a0448023f033c4dda2ef848714b4be2ae45ce598d088fb3efb406e08c5",
        ),
    }
    data = {}
    for subset_name, meta in data_meta.items():
        data[subset_name] = download_and_read_csv(meta, raw_data_cache, header=None)

    n_columns, train_size, test_size = 5000, 6000, 1000

    x_train = convert_x(data["x_train"], train_size, n_columns)
    x_test = convert_x(data["x_test"], test_size, n_columns)
    y_train = convert_y(data["y_train"], train_size)
    y_test = convert_y(data["y_test"], test_size)

    x = np.vstack([x_train, x_test])
    y = np.hstack([y_train, y_test])

    x = StandardScaler(with_mean=True, with_std=True).fit_transform(x)

    data_desc = {
        "n_classes": 2,
        "default_split": {
            "train_size": y_train.shape[0],
            "test_size": y_test.shape[0],
            "shuffle": False,
        },
    }
    return {"x": x, "y": y}, data_desc


@cache
def load_a9a(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    def transform_x_y(x, y):
        y[y == -1] = 0
        return x, y

    x, y = load_openml(1430, raw_data_cache, transform_x_y)
    data_desc = {"n_classes": 2, "default_split": {"test_size": 0.2, "random_state": 11}}
    return {"x": x, "y": y}, data_desc


@cache
def load_codrnanorm(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    def transform_x_y(x, y):
        x = pd.DataFrame(x.todense())
        y = y.astype("int")
        y[y == -1] = 0
        return x, y

    x, y = load_openml(1241, raw_data_cache, transform_x_y_func=transform_x_y)
    data_desc = {"n_classes": 2, "default_split": {"test_size": 0.2, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


@cache
def load_creditcard(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    x, y = load_openml(1597, raw_data_cache)
    data_desc = {"n_classes": 2, "default_split": {"test_size": 0.1, "random_state": 777}}
    return {"x": x, "y": y}, data_desc


@cache
def load_fraud(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    x, y = load_openml(42175, raw_data_cache)
    data_desc = {"n_classes": 2, "default_split": {"test_size": 0.2, "random_state": 77}}
    return {"x": x, "y": y}, data_desc


@cache
def load_ijcnn(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    Author: Danil Prokhorov.
    libSVM,AAD group
    Cite: Danil Prokhorov. IJCNN 2001 neural network competition.
    Slide presentation in IJCNN'01,
    Ford Research Laboratory, 2001. http://www.geocities.com/ijcnn/nnc_ijcnn01.pdf.

    Classification task. n_classes = 2.
    """

    def transform_x_y(x, y):
        y[y == -1] = 0
        return x, y

    x, y = load_openml(1575, raw_data_cache, transform_x_y)
    data_desc = {"n_classes": 2, "default_split": {"test_size": 0.2, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


@cache
def load_klaverjas(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    Abstract:
    Klaverjas is an example of the Jack-Nine card games,
    which are characterized as trick-taking games where the the Jack
    and nine of the trump suit are the highest-ranking trumps, and
    the tens and aces of other suits are the most valuable cards
    of these suits. It is played by four players in two teams.

    Task Information:
    Classification task. n_classes = 2.
    """
    x, y = load_openml(41228, raw_data_cache)
    data_desc = {"n_classes": 2, "default_split": {"train_size": 0.2, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


@cache
def load_skin_segmentation(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    Abstract:
    The Skin Segmentation dataset is constructed over B, G, R color space.
    Skin and Nonskin dataset is generated using skin textures from
    face images of diversity of age, gender, and race people.
    Author: Rajen Bhatt, Abhinav Dhall, rajen.bhatt '@' gmail.com, IIT Delhi.

    Classification task. n_classes = 2.
    """

    def transform_x_y(x, y):
        y = y.astype(int)
        y[y == 2] = 0
        return x, y

    x, y = load_openml(1502, raw_data_cache, transform_x_y)
    data_desc = {"n_classes": 2, "default_split": {"test_size": 0.2, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


@cache
def load_cifar(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    Source:
    University of Toronto
    Collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
    https://www.cs.toronto.edu/~kriz/cifar.html

    Classification task. n_classes = 10.
    """
    x, y = load_openml(40927, raw_data_cache)
    x = StandardScaler(with_mean=True, with_std=False).fit_transform(x)
    binary = dataset_params.get("binary", False)
    if binary:
        y = (y > 0).astype(int)
    data_desc = {
        "n_classes": 2 if binary else 10,
        "default_split": {"test_size": 1 / 6, "random_state": 42},
    }
    return {"x": x, "y": y}, data_desc


@cache
def load_connect(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    Source:
    UC Irvine Machine Learning Repository
    http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.htm

    Classification task. n_classes = 3.
    """
    x, y = load_openml(1591, raw_data_cache)
    y = (y + 1).astype("int")
    data_desc = {"n_classes": 3, "default_split": {"test_size": 0.1, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


@cache
def load_covertype(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    Abstract: This is the original version of the famous
    covertype dataset in ARFF format.
    Author: Jock A. Blackard, Dr. Denis J. Dean, Dr. Charles W. Anderson
    Source: [original](https://archive.ics.uci.edu/ml/datasets/covertype)

    Classification task. n_classes = 7.
    """
    x, y = load_openml(1596, raw_data_cache)
    data_desc = {"n_classes": 7, "default_split": {"test_size": 0.4, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


def load_mnist_template(
    openml_id: int,
    raw_data_cache: str,
) -> Tuple[Dict, Dict]:
    def transform_x_y(x, y):
        return x.astype("uint8"), y.astype("uint8")

    x, y = load_openml(openml_id, raw_data_cache, transform_x_y)
    data_desc = {"n_classes": 10, "default_split": {"test_size": 10000, "shuffle": False}}
    return {"x": x, "y": y}, data_desc


@cache
def load_mnist(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    Abstract:
    The MNIST database of handwritten digits with 784 features.
    It can be split in a training set of the first 60,000 examples,
    and a test set of 10,000 examples
    Source:
    Yann LeCun, Corinna Cortes, Christopher J.C. Burges
    http://yann.lecun.com/exdb/mnist/

    Classification task. n_classes = 10.
    """
    return load_mnist_template(554, raw_data_cache)


@cache
def load_fashion_mnist(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    return load_mnist_template(40996, raw_data_cache)


@cache
def load_svhn(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    return load_mnist_template(41081, raw_data_cache)


@cache
def load_sensit(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    Abstract: Vehicle classification in distributed sensor networks.
    Author: M. Duarte, Y. H. Hu
    Source: [original](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets)

    Classification task. n_classes = 3.
    """
    x, y = load_openml(1593, raw_data_cache)
    data_desc = {"n_classes": 3, "default_split": {"test_size": 0.2, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


@cache
def load_szilard_1m(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    https://github.com/szilard/GBM-perf
    """
    url = "https://s3.amazonaws.com/benchm-ml--main/train-1m.csv"
    d_train = download_and_read_csv(url, raw_data_cache)

    url = "https://s3.amazonaws.com/benchm-ml--main/test.csv"
    d_test = download_and_read_csv(url, raw_data_cache)

    label_col = "dep_delayed_15min"
    y_train = (d_train[label_col] == "Y").astype(int).values
    y_test = (d_test[label_col] == "Y").astype(int).values
    y = np.concatenate([y_train, y_test])

    X_train_raw = d_train.drop(columns=[label_col])
    X_test_raw = d_test.drop(columns=[label_col])

    combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
    X_combined_oh = pd.get_dummies(combined)
    x = sparse.csr_matrix(X_combined_oh.values)

    n_train = len(d_train)
    n_test = len(d_test)
    data_desc = {"default_split": {"test_size": n_train, "test_size": n_test}}

    return {"x": x, "y": y}, data_desc


@cache
def load_szilard_10m(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    https://github.com/szilard/GBM-perf
    """
    url = "https://s3.amazonaws.com/benchm-ml--main/train-10m.csv"
    d_train = download_and_read_csv(url, raw_data_cache)

    url = "https://s3.amazonaws.com/benchm-ml--main/test.csv"
    d_test = download_and_read_csv(url, raw_data_cache)

    label_col = "dep_delayed_15min"
    y_train = (d_train[label_col] == "Y").astype(int).values
    y_test = (d_test[label_col] == "Y").astype(int).values
    y = np.concatenate([y_train, y_test])

    X_train_raw = d_train.drop(columns=[label_col])
    X_test_raw = d_test.drop(columns=[label_col])

    combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
    X_combined_oh = pd.get_dummies(combined, sparse=True)
    x = sparse.csr_matrix(X_combined_oh)

    n_train = len(d_train)
    n_test = len(d_test)
    data_desc = {"default_split": {"test_size": n_train, "test_size": n_test}}

    return {"x": x, "y": y}, data_desc


"""
Regression datasets
"""


@cache
def load_abalone(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    """
    https://archive.ics.uci.edu/ml/machine-learning-databases/abalone

    """

    DATA = (
        (
            "abalone.data",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone",
            "de37cdcdcaaa50c309d514f248f7c2302a5f1f88c168905eba23fe2fbc78449f",
        ),
    )

    data = download_and_read_csv(DATA, raw_data_cache, header=None)
    data[0] = data[0].astype("category").cat.codes
    x, y = data.iloc[:, :-1], data.iloc[:, -1].values

    data_desc = {"default_split": {"test_size": 0.2, "random_state": 0}}
    return {"x": x, "y": y}, data_desc


@cache
def load_california_housing(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    x, y = fetch_california_housing(
        return_X_y=True, as_frame=False, data_home=raw_data_cache
    )
    data_desc = {"default_split": {"test_size": 0.1, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


@cache
def load_fried(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    x, y = load_openml(564, raw_data_cache)
    data_desc = {"default_split": {"test_size": 0.2, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


@cache
def load_medical_charges_nominal(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    x, y = load_openml(42559, raw_data_cache)

    data_desc = {"default_split": {"test_size": 0.2, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


@cache
def load_twodplanes(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    x, y = load_openml(1197, raw_data_cache)
    data_desc = {"default_split": {"test_size": 0.4, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


@cache
def load_year_prediction_msd(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:

    ARCHIVE = (
        (
            "YearPredictionMSD.txt.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00203",
            "06f801af323bb7798e800583acce4ea1ed2697ac12c23f4424aea0a7a3d09e11",
        ),
    )

    data = download_and_read_csv(ARCHIVE, raw_data_cache, header=None)
    x, y = data.iloc[:, 1:], data.iloc[:, 0]
    data_desc = {"default_split": {"test_size": 0.1, "shuffle": False}}
    return {"x": x, "y": y}, data_desc


@cache
def load_yolanda(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:
    x, y = load_openml(42705, raw_data_cache)
    data_desc = {"default_split": {"test_size": 0.2, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


@cache
def load_road_network(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:

    DATA = (
        (
            "3D_spatial_network.txt",
            "http://archive.ics.uci.edu/ml/machine-learning-databases/00246",
            "d83303a61dc3c9d0842df2c7e5b496ec29aafa2080a430253acb8411cae789dc",
        ),
    )

    data = download_and_read_csv(DATA, raw_data_cache, dtype=dtype)

    n_samples, dtype = 20000, np.float32
    x, y = data.values[:, 1:], data.values[:, 0]

    data_desc = {
        "default_split": {
            "train_size": n_samples,
            "test_size": n_samples,
            "shuffle": False,
        }
    }
    return {"x": x, "y": y}, data_desc


"""
Index/neighbors search datasets
"""


def load_ann_dataset_template(DATA: tuple[str, str, str], raw_data_cache):
    import h5py

    local_path = fetch_file(
        url=DATA[1], folder=raw_data_cache, local_filename=DATA[0], sha256=DATA[2]
    )
    with h5py.File(local_path, "r") as f:
        x_train = np.asarray(f["train"])
        x_test = np.asarray(f["test"])
    x = np.concatenate([x_train, x_test], axis=0)
    data_desc = {
        "default_split": {
            "train_size": x_train.shape[0],
            "test_size": x_test.shape[0],
        }
    }
    del x_train, x_test
    # TODO: remove placeholding zeroed y
    y = np.zeros((x.shape[0],))
    return {"x": x, "y": y}, data_desc


@cache
def load_sift(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:

    DATA = (
        "sift-128-euclidean.hdf5",
        "http://ann-benchmarks.com",
        "dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984",
    )

    return load_ann_dataset_template(DATA, raw_data_cache)


@cache
def load_gist(
    data_name: str, data_cache: str, raw_data_cache: str, dataset_params: Dict
) -> Tuple[Dict, Dict]:

    DATA = (
        "gist-960-euclidean.hdf5",
        "http://ann-benchmarks.com",
        "8e95831936bfdbfa0a56086942e2cf98cd703517c67f985914183eb4cdbf026a",
    )

    return load_ann_dataset_template(DATA, raw_data_cache)


dataset_loading_functions = {
    # classification
    "airline_depdelay": load_airline_depdelay,
    "a9a": load_a9a,
    "codrnanorm": load_codrnanorm,
    "covtype": load_covtype,
    "creditcard": load_creditcard,
    "digits": load_sklearn_digits,
    "epsilon": load_epsilon,
    "fraud": load_fraud,
    "gisette": load_gisette,
    "hepmass": load_hepmass,
    "higgs": load_higgs,
    "susy": load_susy,
    "ijcnn": load_ijcnn,
    "klaverjas": load_klaverjas,
    "cifar": load_cifar,
    "connect": load_connect,
    "covertype": load_covertype,
    "skin_segmentation": load_skin_segmentation,
    "mnist": load_mnist,
    "fashion_mnist": load_fashion_mnist,
    "svhn": load_svhn,
    "sensit": load_sensit,
    "letters": load_letters,
    "szilard_1m": load_szilard_1m,
    "szilard_10m": load_szilard_10m,
    # regression
    "abalone": load_abalone,
    "california_housing": load_california_housing,
    "fried": load_fried,
    "medical_charges_nominal": load_medical_charges_nominal,
    "twodplanes": load_twodplanes,
    "year_prediction_msd": load_year_prediction_msd,
    "yolanda": load_yolanda,
    "road_network": load_road_network,
    # index search
    "sift": load_sift,
    "gist": load_gist,
}
dataset_loading_functions = {
    key: preprocess(value) for key, value in dataset_loading_functions.items()
}
