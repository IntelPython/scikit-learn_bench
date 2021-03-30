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
import re
import tarfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import tqdm
from sklearn.datasets import fetch_covtype, fetch_openml, load_svmlight_file
from sklearn.model_selection import train_test_split

pbar: tqdm.tqdm = None


def _show_progress(block_num: int, block_size: int, total_size: int) -> None:
    global pbar
    if pbar is None:
        pbar = tqdm.tqdm(total=total_size / 1024, unit='kB')

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(block_size / 1024)
    else:
        pbar.close()
        pbar = None


def _retrieve(url: str, filename: str) -> None:
    urlretrieve(url, filename, reporthook=_show_progress)


def _read_libsvm_msrank(file_obj, n_samples, n_features, dtype):
    X = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples,))

    counter = 0

    regexp = re.compile(r'[A-Za-z0-9]+:(-?\d*\.?\d+)')

    for line in file_obj:
        line = str(line).replace("\\n'", "")
        line = regexp.sub(r'\g<1>', line)
        line = line.rstrip(" \n\r").split(' ')

        y[counter] = int(line[0])
        X[counter] = [float(i) for i in line[1:]]

        counter += 1
        if counter == n_samples:
            break

    return np.array(X, dtype=dtype), np.array(y, dtype=dtype)


def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def _count_lines(filename):
    with open(filename, 'rb') as f:
        f_gen = _make_gen(f.read)
        return sum(buf.count(b'\n') for buf in f_gen)


def a_nine_a(dataset_dir: Path) -> bool:
    """
    Author: Ronny Kohavi","Barry Becker
    libSVM","AAD group
    Source: original - Date unknown
    Cite: http://archive.ics.uci.edu/ml/datasets/Adult

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
    dataset_name = 'airline'
    os.makedirs(dataset_dir, exist_ok=True)

    url = 'http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2'
    local_url = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        _retrieve(url, local_url)
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
        logging.info(f'Started loading {dataset_name}')
        _retrieve(url_train, local_url_train)
    if not os.path.isfile(local_url_test):
        logging.info(f'Started loading {dataset_name}')
        _retrieve(url_test, local_url_test)
    logging.info(f'{dataset_name} is loaded, started parsing...')

    sets = []
    labels = []

    categorical_names = ["Month", "DayofMonth",
                         "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

    for local_url in [local_url_train, local_url_train]:
        df = pd.read_csv(local_url, nrows=1000000
                         if local_url.endswith('train-10m.csv') else None)
        X = df.drop('dep_delayed_15min', 1)
        y = df["dep_delayed_15min"]

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


def connect(dataset_dir: Path) -> bool:
    """
    Source:
    UC Irvine Machine Learning Repository
    http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.htm

    Classification task. n_classes = 3.
    connect X train dataset (196045, 127)
    connect y train dataset (196045, 1)
    connect X test dataset  (49012,  127)
    connect y test dataset  (49012,  1)
    """
    dataset_name = 'connect'
    os.makedirs(dataset_dir, exist_ok=True)

    X, y = fetch_openml(name='connect-4', return_X_y=True,
                        as_frame=False, data_home=dataset_dir)
    X = pd.DataFrame(X.todense())
    y = pd.DataFrame(y)
    y = y.astype(int)

    logging.info(f'{dataset_name} dataset is downloaded')
    logging.info('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)
    for data, name in zip((x_train, x_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.csv'
        data.to_csv(os.path.join(dataset_dir, filename),
                    header=False, index=False)
    logging.info(f'dataset {dataset_name} ready.')
    return True


def covertype(dataset_dir: Path) -> bool:
    """
    Abstract: This is the original version of the famous
    covertype dataset in ARFF format.
    Author: Jock A. Blackard, Dr. Denis J. Dean, Dr. Charles W. Anderson
    Source: [original](https://archive.ics.uci.edu/ml/datasets/covertype)

    Classification task. n_classes = 7.
    covertype X train dataset (390852, 54)
    covertype y train dataset (390852, 1)
    covertype X test dataset  (97713,  54)
    covertype y test dataset  (97713,  1)
    """
    dataset_name = 'covertype'
    os.makedirs(dataset_dir, exist_ok=True)

    X, y = fetch_openml(name='covertype', version=3, return_X_y=True,
                        as_frame=True, data_home=dataset_dir)
    y = y.astype(int)

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


def covtype(dataset_dir: Path) -> bool:
    dataset_name = 'covtype'
    os.makedirs(dataset_dir, exist_ok=True)

    logging.info(f'Started loading {dataset_name}')
    X, y = fetch_covtype(return_X_y=True)  # pylint: disable=unexpected-keyword-arg
    logging.info(f'{dataset_name} is loaded, started parsing...')

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    for data, name in zip((X_train, X_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def epsilon(dataset_dir: Path) -> bool:
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
        _retrieve(url_train, local_url_train)
    if not os.path.isfile(local_url_test):
        logging.info(f'Started loading {dataset_name}, test')
        _retrieve(url_test, local_url_test)
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
        _retrieve(gisette_train_data_url, filename_train_data)

    gisette_train_labels_url = domen_hhtp + '/gisette/GISETTE/gisette_train.labels'
    filename_train_labels = os.path.join(cache_dir, 'gisette_train.labels')
    if not os.path.exists(filename_train_labels):
        _retrieve(gisette_train_labels_url, filename_train_labels)

    gisette_test_data_url = domen_hhtp + '/gisette/GISETTE/gisette_valid.data'
    filename_test_data = os.path.join(cache_dir, 'gisette_valid.data')
    if not os.path.exists(filename_test_data):
        _retrieve(gisette_test_data_url, filename_test_data)

    gisette_test_labels_url = domen_hhtp + '/gisette/gisette_valid.labels'
    filename_test_labels = os.path.join(cache_dir, 'gisette_valid.labels')
    if not os.path.exists(filename_test_labels):
        _retrieve(gisette_test_labels_url, filename_test_labels)

    logging.info('gisette dataset is downloaded')
    logging.info('reading CSV file...')

    num_cols = 5000

    df_train = pd.read_csv(filename_train_data, header=None)
    df_labels = pd.read_csv(filename_train_labels, header=None)
    num_train = 6000
    x_train_arr = df_train.iloc[:num_train].values
    x_train = pd.DataFrame(np.array([np.fromstring(
        elem[0], dtype=int, count=num_cols, sep=' ') for elem in x_train_arr]))
    y_train_arr = df_labels.iloc[:num_train].values
    y_train = pd.DataFrame((y_train_arr > 0).astype(int))

    num_train = 1000
    df_test = pd.read_csv(filename_test_data, header=None)
    df_labels = pd.read_csv(filename_test_labels, header=None)
    x_test_arr = df_test.iloc[:num_train].values
    x_test = pd.DataFrame(np.array(
        [np.fromstring(
            elem[0],
            dtype=int, count=num_cols, sep=' ')
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
    dataset_name = 'higgs'
    os.makedirs(dataset_dir, exist_ok=True)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
    local_url = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        _retrieve(url, local_url)
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
    Higgs dataset from UCI machine learning repository (
    https://archive.ics.uci.edu/ml/datasets/HIGGS).
    TaskType:binclass
    NumberOfFeatures:28
    NumberOfInstances:11M
    """
    dataset_name = 'higgs1m'
    os.makedirs(dataset_dir, exist_ok=True)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
    local_url = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        _retrieve(url, local_url)
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


def mnist(dataset_dir: Path) -> bool:
    """
    Abstract:
    The MNIST database of handwritten digits with 784 features.
    It can be split in a training set of the first 60,000 examples,
    and a test set of 10,000 examples
    Source:
    Yann LeCun, Corinna Cortes, Christopher J.C. Burges
    http://yann.lecun.com/exdb/mnist/

    Classification task. n_classes = 10.
    mnist X train dataset (60000, 784)
    mnist y train dataset (60000, 1)
    mnist X test dataset  (10000,  784)
    mnist y test dataset  (10000,  1)
    """
    dataset_name = 'mnist'

    os.makedirs(dataset_dir, exist_ok=True)

    X, y = fetch_openml(name='mnist_784', return_X_y=True,
                        as_frame=True, data_home=dataset_dir)
    y = y.astype(int)
    X = X / 255

    logging.info(f'{dataset_name} dataset is downloaded')
    logging.info('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, shuffle=False)
    for data, name in zip((x_train, x_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.csv'
        data.to_csv(os.path.join(dataset_dir, filename),
                    header=False, index=False)
    logging.info(f'dataset {dataset_name} ready.')
    return True


def mortgage_first_q(dataset_dir: Path) -> bool:
    return False


def msrank(dataset_dir: Path) -> bool:
    """
    Dataset from szilard benchmarks: https://github.com/szilard/GBM-perf
    TaskType:binclass
    NumberOfFeatures:700
    NumberOfInstances:10100000
    """
    dataset_name = 'msrank'
    os.makedirs(dataset_dir, exist_ok=True)
    url = "https://storage.mds.yandex.net/get-devtools-opensource/471749/msrank.tar.gz"
    local_url = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        _retrieve(url, local_url)
        logging.info(f'{dataset_name} is loaded, unzipping...')
        tar = tarfile.open(local_url, "r:gz")
        tar.extractall(dataset_dir)
        tar.close()
        logging.info(f'{dataset_name} is unzipped, started parsing...')

    sets = []
    labels = []
    n_features = 137

    for set_name in ['train.txt', 'vali.txt', 'test.txt']:
        file_name = str(dataset_dir) + os.path.join('MSRank', set_name)

        n_samples = _count_lines(file_name)
        with open(file_name, 'r') as file_obj:
            X, y = _read_libsvm_msrank(file_obj, n_samples, n_features, np.float32)

        sets.append(X)
        labels.append(y)

    sets[0] = np.vstack((sets[0], sets[1]))
    labels[0] = np.hstack((labels[0], labels[1]))

    sets = [np.ascontiguousarray(sets[i]) for i in [0, 2]]
    labels = [np.ascontiguousarray(labels[i]) for i in [0, 2]]

    for data, name in zip((sets[0], sets[1], labels[0], labels[1]),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def plasticc(dataset_dir: Path) -> bool:
    return False


def santander(dataset_dir: Path) -> bool:
    return False


def sensit(dataset_dir: Path) -> bool:
    """
    Abstract: Vehicle classification in distributed sensor networks.
    Author: M. Duarte, Y. H. Hu
    Source: [original](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets)

    Classification task. n_classes = 2.
    sensit X train dataset (196045, 3)
    sensit y train dataset (196045, 1)
    sensit X test dataset  (49012,  3)
    sensit y test dataset  (49012,  1)
    """
    dataset_name = 'sensit'
    os.makedirs(dataset_dir, exist_ok=True)

    X, y = fetch_openml(name='SensIT-Vehicle-Combined',
                        return_X_y=True, as_frame=False, data_home=dataset_dir)
    X = pd.DataFrame(X.todense())
    y = pd.DataFrame(y)
    y = y.astype(int)

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


def year(dataset_dir: Path) -> bool:
    dataset_name = 'year'
    os.makedirs(dataset_dir, exist_ok=True)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt' \
          '.zip'
    local_url = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        _retrieve(url, local_url)
    logging.info(f'{dataset_name} is loaded, started parsing...')

    year = pd.read_csv(local_url, header=None)
    X = year.iloc[:, 1:].to_numpy(dtype=np.float32)
    y = year.iloc[:, 0].to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False,
                                                        train_size=463715,
                                                        test_size=51630)

    for data, name in zip((X_train, X_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True
