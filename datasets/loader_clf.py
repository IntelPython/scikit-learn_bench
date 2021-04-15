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

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_svmlight_file
from sklearn.model_selection import train_test_split

from .loader_utils import retrieve


def a_nine_a(dataset_dir: Path) -> bool:
    """
    Author: Ronny Kohavi","Barry Becker
    libSVM","AAD group
    Source: original - Date unknown
    Site: http://archive.ics.uci.edu/ml/datasets/Adult

    Classification task. n_classes = 2.
    a9a X train dataset (39073, 123)
    a9a y train dataset (39073, 1)
    a9a X test dataset  (9769,  123)
    a9a y test dataset  (9769,  1)
    """
    dataset_name = 'a9a'
    os.makedirs(dataset_dir, exist_ok=True)

    X, y = fetch_openml(name='a9a', return_X_y=True,
                        as_frame=False, data_home=dataset_dir)
    X = pd.DataFrame(X.todense())
    y = pd.DataFrame(y)

    y[y == -1] = 0

    logging.info('a9a dataset is downloaded')
    logging.info('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=11)
    for data, name in zip((x_train, x_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.csv'
        data.to_csv(os.path.join(dataset_dir, filename),
                    header=False, index=False)
    logging.info(f'dataset {dataset_name} ready.')
    return True


def airline(dataset_dir: Path) -> bool:
    """
    Airline dataset
    http://kt.ijs.si/elena_ikonomovska/data.html

    TaskType:binclass
    NumberOfFeatures:13
    NumberOfInstances:115M
    """
    dataset_name = 'airline'
    os.makedirs(dataset_dir, exist_ok=True)

    url = 'http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2'
    local_url = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        retrieve(url, local_url)
    logging.info(f'{dataset_name} is loaded, started parsing...')

    cols = [
        "Year", "Month", "DayofMonth", "DayofWeek", "CRSDepTime",
        "CRSArrTime", "UniqueCarrier", "FlightNum", "ActualElapsedTime",
        "Origin", "Dest", "Distance", "Diverted", "ArrDelay"
    ]

    # load the data as int16
    dtype = np.int16

    dtype_columns = {
        "Year": dtype, "Month": dtype, "DayofMonth": dtype, "DayofWeek": dtype,
        "CRSDepTime": dtype, "CRSArrTime": dtype, "FlightNum": dtype,
        "ActualElapsedTime": dtype, "Distance":
            dtype,
        "Diverted": dtype, "ArrDelay": dtype,
    }

    df: Any = pd.read_csv(local_url, names=cols, dtype=dtype_columns)

    # Encode categoricals as numeric
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].astype("category").cat.codes

    # Turn into binary classification problem
    df["ArrDelayBinary"] = 1 * (df["ArrDelay"] > 0)

    X = df[df.columns.difference(["ArrDelay", "ArrDelayBinary"])
           ].to_numpy(dtype=np.float32)
    y = df["ArrDelayBinary"].to_numpy(dtype=np.float32)
    del df
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    for data, name in zip((X_train, X_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def airline_ohe(dataset_dir: Path) -> bool:
    """
    Dataset from szilard benchmarks: https://github.com/szilard/GBM-perf
    TaskType:binclass
    NumberOfFeatures:700
    NumberOfInstances:10100000
    """
    dataset_name = 'airline-ohe'
    os.makedirs(dataset_dir, exist_ok=True)

    url_train = 'https://s3.amazonaws.com/benchm-ml--main/train-10m.csv'
    url_test = 'https://s3.amazonaws.com/benchm-ml--main/test.csv'
    local_url_train = os.path.join(dataset_dir, os.path.basename(url_train))
    local_url_test = os.path.join(dataset_dir, os.path.basename(url_test))
    if not os.path.isfile(local_url_train):
        logging.info(f'Started loading {dataset_name} train')
        retrieve(url_train, local_url_train)
    if not os.path.isfile(local_url_test):
        logging.info(f'Started loading {dataset_name} test')
        retrieve(url_test, local_url_test)
    logging.info(f'{dataset_name} is loaded, started parsing...')

    sets = []
    labels = []

    categorical_names = ["Month", "DayofMonth",
                         "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

    for local_url in [local_url_train, local_url_train]:
        df = pd.read_csv(local_url, nrows=1000000
                         if local_url.endswith('train-10m.csv') else None)
        X = df.drop('dep_delayed_15min', 1)
        y: Any = df["dep_delayed_15min"]

        y_num = np.where(y == "Y", 1, 0)

        sets.append(X)
        labels.append(y_num)

    n_samples_train = sets[0].shape[0]

    X_final: Any = pd.concat(sets)
    X_final = pd.get_dummies(X_final, columns=categorical_names)
    sets = [X_final[:n_samples_train], X_final[n_samples_train:]]

    for data, name in zip((sets[0], sets[1], labels[0], labels[1]),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def bosch(dataset_dir: Path) -> bool:
    """
    Bosch Production Line Performance data set
    https://www.kaggle.com/c/bosch-production-line-performance

    Requires Kaggle API and API token (https://github.com/Kaggle/kaggle-api)
    Contains missing values as NaN.

    TaskType:binclass
    NumberOfFeatures:968
    NumberOfInstances:1.184M
    """
    dataset_name = 'bosch'
    os.makedirs(dataset_dir, exist_ok=True)

    filename = "train_numeric.csv.zip"
    local_url = os.path.join(dataset_dir, filename)

    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        os.system(
            "kaggle competitions download -c bosch-production-line-performance -f " +
            filename + " -p " + str(dataset_dir))
    logging.info(f'{dataset_name} is loaded, started parsing...')
    X = pd.read_csv(local_url, index_col=0, compression='zip', dtype=np.float32)
    y = X.iloc[:, -1].to_numpy(dtype=np.float32)
    X.drop(X.columns[-1], axis=1, inplace=True)
    X_np = X.to_numpy(dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X_np, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    for data, name in zip((X_train, X_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def codrnanorm(dataset_dir: Path) -> bool:
    """
    Abstract: Detection of non-coding RNAs on the basis of predicted secondary
    structure formation free energy change.
    Author: Andrew V Uzilov,Joshua M Keegan,David H Mathews.
    Source: [original](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets)

    Classification task. n_classes = 2.
    codrnanorm X train dataset (390852, 8)
    codrnanorm y train dataset (390852, 1)
    codrnanorm X test dataset  (97713,  8)
    codrnanorm y test dataset  (97713,  1)
    """
    dataset_name = 'codrnanorm'
    os.makedirs(dataset_dir, exist_ok=True)

    X, y = fetch_openml(name='codrnaNorm', return_X_y=True,
                        as_frame=False, data_home=dataset_dir)
    X = pd.DataFrame(X.todense())
    y = pd.DataFrame(y)

    logging.info(f'{dataset_name} dataset is downloaded')
    logging.info('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    for data, name in zip((x_train, x_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.csv'
        data.to_csv(os.path.join(dataset_dir, filename),
                    header=False, index=False)
    logging.info(f'dataset {dataset_name} ready.')
    return True


def epsilon(dataset_dir: Path) -> bool:
    """
    Epsilon dataset
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

    TaskType:binclass
    NumberOfFeatures:2000
    NumberOfInstances:500K
    """
    dataset_name = 'epsilon'
    os.makedirs(dataset_dir, exist_ok=True)

    url_train = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary' \
                '/epsilon_normalized.bz2'
    url_test = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary' \
               '/epsilon_normalized.t.bz2'
    local_url_train = os.path.join(dataset_dir, os.path.basename(url_train))
    local_url_test = os.path.join(dataset_dir, os.path.basename(url_test))

    if not os.path.isfile(local_url_train):
        logging.info(f'Started loading {dataset_name}, train')
        retrieve(url_train, local_url_train)
    if not os.path.isfile(local_url_test):
        logging.info(f'Started loading {dataset_name}, test')
        retrieve(url_test, local_url_test)
    logging.info(f'{dataset_name} is loaded, started parsing...')
    X_train, y_train = load_svmlight_file(local_url_train,
                                          dtype=np.float32)
    X_test, y_test = load_svmlight_file(local_url_test,
                                        dtype=np.float32)
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    y_train[y_train <= 0] = 0
    y_test[y_test <= 0] = 0

    for data, name in zip((X_train, X_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def fraud(dataset_dir: Path) -> bool:
    """
    Credit Card Fraud Detection contest
    https://www.kaggle.com/mlg-ulb/creditcardfraud

    Requires Kaggle API and API token (https://github.com/Kaggle/kaggle-api)
    Contains missing values as NaN.

    TaskType:binclass
    NumberOfFeatures:30
    NumberOfInstances:285K
    """
    dataset_name = 'fraud'
    os.makedirs(dataset_dir, exist_ok=True)

    filename = "creditcard.csv"
    local_url = os.path.join(dataset_dir, filename)

    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        os.system("kaggle datasets download mlg-ulb/creditcardfraud -f" +
                  filename + " -p " + str(dataset_dir))
    logging.info(f'{dataset_name} is loaded, started parsing...')

    df = pd.read_csv(local_url + ".zip", dtype=np.float32)
    X = df[[col for col in df.columns if col.startswith('V')]].to_numpy(dtype=np.float32)
    y = df['Class'].to_numpy(dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    for data, name in zip((X_train, X_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def gisette(dataset_dir: Path) -> bool:
    """
    GISETTE is a handwritten digit recognition problem.
    The problem is to separate the highly confusable digits '4' and '9'.
    This dataset is one of five datasets of the NIPS 2003 feature selection challenge.

    Classification task. n_classes = 2.
    gisette X train dataset (6000, 5000)
    gisette y train dataset (6000, 1)
    gisette X test dataset  (1000, 5000)
    gisette y test dataset  (1000, 1)
    """
    dataset_name = 'gisette'
    os.makedirs(dataset_dir, exist_ok=True)

    cache_dir = os.path.join(dataset_dir, '_gisette')
    os.makedirs(cache_dir, exist_ok=True)

    domen_hhtp = 'http://archive.ics.uci.edu/ml/machine-learning-databases/'

    gisette_train_data_url = domen_hhtp + '/gisette/GISETTE/gisette_train.data'
    filename_train_data = os.path.join(cache_dir, 'gisette_train.data')
    if not os.path.exists(filename_train_data):
        retrieve(gisette_train_data_url, filename_train_data)

    gisette_train_labels_url = domen_hhtp + '/gisette/GISETTE/gisette_train.labels'
    filename_train_labels = os.path.join(cache_dir, 'gisette_train.labels')
    if not os.path.exists(filename_train_labels):
        retrieve(gisette_train_labels_url, filename_train_labels)

    gisette_test_data_url = domen_hhtp + '/gisette/GISETTE/gisette_valid.data'
    filename_test_data = os.path.join(cache_dir, 'gisette_valid.data')
    if not os.path.exists(filename_test_data):
        retrieve(gisette_test_data_url, filename_test_data)

    gisette_test_labels_url = domen_hhtp + '/gisette/gisette_valid.labels'
    filename_test_labels = os.path.join(cache_dir, 'gisette_valid.labels')
    if not os.path.exists(filename_test_labels):
        retrieve(gisette_test_labels_url, filename_test_labels)

    logging.info('gisette dataset is downloaded')
    logging.info('reading CSV file...')

    num_cols = 5000

    df_train = pd.read_csv(filename_train_data, header=None)
    df_labels = pd.read_csv(filename_train_labels, header=None)
    num_train = 6000
    x_train_arr: Any = df_train.iloc[:num_train].values
    x_train = pd.DataFrame(np.array([np.fromstring(
        elem[0], dtype=int, count=num_cols, sep=' ').tolist() for elem in x_train_arr]))
    y_train_arr = df_labels.iloc[:num_train].values
    y_train = pd.DataFrame((y_train_arr > 0).astype(int))

    num_train = 1000
    df_test = pd.read_csv(filename_test_data, header=None)
    df_labels = pd.read_csv(filename_test_labels, header=None)
    x_test_arr: Any = df_test.iloc[:num_train].values
    x_test = pd.DataFrame(np.array(
        [np.fromstring(
            elem[0],
            dtype=int, count=num_cols, sep=' ').tolist()
         for elem in x_test_arr]))
    y_test_arr = df_labels.iloc[:num_train].values
    y_test = pd.DataFrame((y_test_arr > 0).astype(int))

    for data, name in zip((x_train, x_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.csv'
        data.to_csv(os.path.join(dataset_dir, filename),
                    header=False, index=False)

    logging.info('dataset gisette ready.')
    return True


def higgs(dataset_dir: Path) -> bool:
    """
    Higgs dataset from UCI machine learning repository
    https://archive.ics.uci.edu/ml/datasets/HIGGS

    TaskType:binclass
    NumberOfFeatures:28
    NumberOfInstances:11M
    """
    dataset_name = 'higgs'
    os.makedirs(dataset_dir, exist_ok=True)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
    local_url = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        retrieve(url, local_url)
    logging.info(f'{dataset_name} is loaded, started parsing...')

    higgs = pd.read_csv(local_url)
    X = higgs.iloc[:, 1:].to_numpy(dtype=np.float32)
    y = higgs.iloc[:, 0].to_numpy(dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    for data, name in zip((X_train, X_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def higgs_one_m(dataset_dir: Path) -> bool:
    """
    Higgs dataset from UCI machine learning repository
    https://archive.ics.uci.edu/ml/datasets/HIGGS

    Only first 1.5M samples is taken

    TaskType:binclass
    NumberOfFeatures:28
    NumberOfInstances:1.5M
    """
    dataset_name = 'higgs1m'
    os.makedirs(dataset_dir, exist_ok=True)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
    local_url = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        retrieve(url, local_url)
    logging.info(f'{dataset_name} is loaded, started parsing...')

    nrows_train, nrows_test, dtype = 1000000, 500000, np.float32
    data: Any = pd.read_csv(local_url, delimiter=",", header=None,
                            compression="gzip", dtype=dtype, nrows=nrows_train+nrows_test)

    data = data[list(data.columns[1:])+list(data.columns[0:1])]
    n_features = data.shape[1]-1
    train_data = np.ascontiguousarray(data.values[:nrows_train, :n_features], dtype=dtype)
    train_label = np.ascontiguousarray(data.values[:nrows_train, n_features], dtype=dtype)
    test_data = np.ascontiguousarray(
        data.values[nrows_train: nrows_train + nrows_test, : n_features],
        dtype=dtype)
    test_label = np.ascontiguousarray(
        data.values[nrows_train: nrows_train + nrows_test, n_features],
        dtype=dtype)
    for data, name in zip((train_data, test_data, train_label, test_label),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def ijcnn(dataset_dir: Path) -> bool:
    """
    Author: Danil Prokhorov.
    libSVM,AAD group
    Cite: Danil Prokhorov. IJCNN 2001 neural network competition.
    Slide presentation in IJCNN'01,
    Ford Research Laboratory, 2001. http://www.geocities.com/ijcnn/nnc_ijcnn01.pdf.

    Classification task. n_classes = 2.
    ijcnn X train dataset (153344, 22)
    ijcnn y train dataset (153344, 1)
    ijcnn X test dataset  (38337,  22)
    ijcnn y test dataset  (38337,  1)
    """
    dataset_name = 'ijcnn'
    os.makedirs(dataset_dir, exist_ok=True)

    X, y = fetch_openml(name='ijcnn', return_X_y=True,
                        as_frame=False, data_home=dataset_dir)
    X = pd.DataFrame(X.todense())
    y = pd.DataFrame(y)

    y[y == -1] = 0

    logging.info(f'{dataset_name} dataset is downloaded')
    logging.info('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    for data, name in zip((x_train, x_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.csv'
        data.to_csv(os.path.join(dataset_dir, filename),
                    header=False, index=False)
    logging.info(f'dataset {dataset_name} ready.')
    return True


def klaverjas(dataset_dir: Path) -> bool:
    """
    Abstract:
    Klaverjas is an example of the Jack-Nine card games,
    which are characterized as trick-taking games where the the Jack
    and nine of the trump suit are the highest-ranking trumps, and
    the tens and aces of other suits are the most valuable cards
    of these suits. It is played by four players in two teams.

    Task Information:
    Classification task. n_classes = 2.
    klaverjas X train dataset (196045, 3)
    klaverjas y train dataset (196045, 1)
    klaverjas X test dataset  (49012,  3)
    klaverjas y test dataset  (49012,  1)
    """
    dataset_name = 'klaverjas'
    os.makedirs(dataset_dir, exist_ok=True)

    X, y = fetch_openml(name='Klaverjas2018', return_X_y=True,
                        as_frame=True, data_home=dataset_dir)

    y = y.cat.codes
    logging.info(f'{dataset_name} dataset is downloaded')
    logging.info('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.2, random_state=42)
    for data, name in zip((x_train, x_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.csv'
        data.to_csv(os.path.join(dataset_dir, filename),
                    header=False, index=False)
    logging.info(f'dataset {dataset_name} ready.')
    return True


def santander(dataset_dir: Path) -> bool:
    """
    Still doesn't have an loading instruction
    """
    return False


def skin_segmentation(dataset_dir: Path) -> bool:
    """
    Abstract:
    The Skin Segmentation dataset is constructed over B, G, R color space.
    Skin and Nonskin dataset is generated using skin textures from
    face images of diversity of age, gender, and race people.
    Author: Rajen Bhatt, Abhinav Dhall, rajen.bhatt '@' gmail.com, IIT Delhi.

    Classification task. n_classes = 2.
    skin_segmentation X train dataset (196045, 3)
    skin_segmentation y train dataset (196045, 1)
    skin_segmentation X test dataset  (49012,  3)
    skin_segmentation y test dataset  (49012,  1)
    """
    dataset_name = 'skin_segmentation'
    os.makedirs(dataset_dir, exist_ok=True)

    X, y = fetch_openml(name='skin-segmentation',
                        return_X_y=True, as_frame=True, data_home=dataset_dir)
    y = y.astype(int)
    y[y == 2] = 0

    logging.info(f'{dataset_name} dataset is downloaded')
    logging.info('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    for data, name in zip((x_train, x_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.csv'
        data.to_csv(os.path.join(dataset_dir, filename),
                    header=False, index=False)
    logging.info(f'dataset {dataset_name} ready.')
    return True
