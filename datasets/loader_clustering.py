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


def epsilon_50K_cluster(dataset_dir: Path) -> bool:
    """
    Epsilon dataset
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

    Clustering task. n_classes = 2.
    epsilon_50K x cluster dataset (50000, 2001)
    """
    dataset_name = 'epsilon_50K_cluster'
    os.makedirs(dataset_dir, exist_ok=True)

    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary' \
          '/epsilon_normalized.bz2'
    local_url = os.path.join(dataset_dir, os.path.basename(url))

    num_train = 50000
    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        retrieve(url, local_url)
    logging.info(f'{dataset_name} is loaded, started parsing...')
    x_train, y_train = load_svmlight_file(local_url,
                                          dtype=np.float32)

    x_train = x_train.toarray()[:num_train]
    y_train = y_train[:num_train]
    y_train[y_train <= 0] = 0

    filename = f'{dataset_name}.npy'
    data = np.concatenate((x_train, y_train[:, None]), axis=1)
    np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def cifar_cluster(dataset_dir: Path) -> bool:
    """
    Cifar dataset from LIBSVM Datasets (
    https://www.cs.toronto.edu/~kriz/cifar.html#cifar)
    TaskType: Clustering
    cifar x cluster dataset (50000, 3073)
    """
    dataset_name = 'cifar_cluster'
    os.makedirs(dataset_dir, exist_ok=True)

    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/cifar10.bz2'
    local_url = os.path.join(dataset_dir, os.path.basename(url))

    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        retrieve(url, local_url)
    logging.info(f'{dataset_name} is loaded, started parsing...')
    x_train, y_train = load_svmlight_file(local_url,
                                          dtype=np.float32)

    x_train = x_train.toarray()
    y_train = (y_train > 0).astype(int)

    filename = f'{dataset_name}.npy'
    data = np.concatenate((x_train, y_train[:, None]), axis=1)
    np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def higgs_one_m_clustering(dataset_dir: Path) -> bool:
    """
    Higgs dataset from UCI machine learning repository
    https://archive.ics.uci.edu/ml/datasets/HIGGS

    Clustering task. n_classes = 2.
    higgs1m X cluster dataset (1000000, 29)
    """
    dataset_name = 'higgs_one_m_clustering'
    os.makedirs(dataset_dir, exist_ok=True)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
    local_url = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        retrieve(url, local_url)
    logging.info(f'{dataset_name} is loaded, started parsing...')

    nrows_train, nrows_test, dtype = 1000000, 500000, np.float32
    data: Any = pd.read_csv(local_url, delimiter=",", header=None,
                            compression="gzip", dtype=dtype,
                            nrows=nrows_train + nrows_test)

    X = data[data.columns[1:]]
    y = data[data.columns[0:1]]

    x_train, _, y_train, _ = train_test_split(
        X, y, train_size=nrows_train, test_size=nrows_test, shuffle=False)

    filename = f'{dataset_name}.npy'
    data = np.concatenate((x_train, y_train), axis=1)
    np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def hepmass_1M_cluster(dataset_dir: Path) -> bool:
    """
    HEPMASS dataset from UCI machine learning repository (
    https://archive.ics.uci.edu/ml/datasets/HEPMASS).

    Clustering task. n_classes = 2.
    hepmass_10K X cluster dataset (1000000, 29)
    """
    dataset_name = 'hepmass_1M_cluster'
    os.makedirs(dataset_dir, exist_ok=True)

    url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00347/all_train.csv.gz'

    local_url_train = os.path.join(dataset_dir, os.path.basename(url_train))

    if not os.path.isfile(local_url_train):
        logging.info(f'Started loading {dataset_name}, train')
        retrieve(url_train, local_url_train)
    logging.info(f'{dataset_name} is loaded, started parsing...')

    nrows_train, dtype = 1000000, np.float32
    data_train: Any = pd.read_csv(local_url_train, delimiter=",",
                                  compression="gzip", dtype=dtype,
                                  nrows=nrows_train)

    x_train = np.ascontiguousarray(data_train.values[:nrows_train, 1:], dtype=dtype)
    y_train = np.ascontiguousarray(data_train.values[:nrows_train, 0], dtype=dtype)

    filename = f'{dataset_name}.npy'
    data = np.concatenate((x_train, y_train[:, None]), axis=1)
    np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def hepmass_10K_cluster(dataset_dir: Path) -> bool:
    """
    HEPMASS dataset from UCI machine learning repository (
    https://archive.ics.uci.edu/ml/datasets/HEPMASS).

    Clustering task. n_classes = 2.
    hepmass_10K X cluster dataset (10000, 29)
    """
    dataset_name = 'hepmass_10K_cluster'
    os.makedirs(dataset_dir, exist_ok=True)

    url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00347/all_train.csv.gz'

    local_url_train = os.path.join(dataset_dir, os.path.basename(url_train))

    if not os.path.isfile(local_url_train):
        logging.info(f'Started loading {dataset_name}, train')
        retrieve(url_train, local_url_train)
    logging.info(f'{dataset_name} is loaded, started parsing...')

    nrows_train, dtype = 10000, np.float32
    data_train: Any = pd.read_csv(local_url_train, delimiter=",",
                                  compression="gzip", dtype=dtype,
                                  nrows=nrows_train)

    x_train = np.ascontiguousarray(data_train.values[:nrows_train, 1:], dtype=dtype)
    y_train = np.ascontiguousarray(data_train.values[:nrows_train, 0], dtype=dtype)

    filename = f'{dataset_name}.npy'
    data = np.concatenate((x_train, y_train[:, None]), axis=1)
    np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def susy_cluster(dataset_dir: Path) -> bool:
    """
    SUSY dataset from UCI machine learning repository (
    https://archive.ics.uci.edu/ml/datasets/SUSY).

    Clustering task. n_classes = 2.
    susy X cluster dataset (4500000, 29)
    """
    dataset_name = 'susy_cluster'
    os.makedirs(dataset_dir, exist_ok=True)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz'
    local_url = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        retrieve(url, local_url)
    logging.info(f'{dataset_name} is loaded, started parsing...')

    nrows_train, dtype = 4500000, np.float32
    data_raw: Any = pd.read_csv(local_url, delimiter=",", header=None,
                                compression="gzip", dtype=dtype,
                                nrows=nrows_train)

    X = data_raw.iloc[:nrows_train, 1:].values
    y = data_raw.iloc[:nrows_train, 0].values
    data = np.concatenate((X, y[:, None]), axis=1)

    filename = f'{dataset_name}.npy'
    np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def mnist_10K_cluster(dataset_dir: Path) -> bool:
    """
    Abstract:
    The MNIST database of handwritten digits with 784 features.
    It can be split in a training set of the first 60,000 examples,
    and a test set of 10,000 examples
    Source:
    Yann LeCun, Corinna Cortes, Christopher J.C. Burges
    http://yann.lecun.com/exdb/mnist/

    Clustering task. n_classes = 10.
    mnist x clustering dataset  (10000,  785)
    """
    dataset_name = 'mnist_10K_cluster'

    os.makedirs(dataset_dir, exist_ok=True)

    nrows_train, dtype = 10000, np.float32
    X, y = fetch_openml(name='mnist_784', return_X_y=True,
                        as_frame=True, data_home=dataset_dir)
    y = y.astype(int)
    logging.info(f'{dataset_name} is loaded, started parsing...')

    x_train = np.ascontiguousarray(X.values[:nrows_train, 1:], dtype=dtype)
    y_train = np.ascontiguousarray(y.values[:nrows_train], dtype=dtype)

    filename = f'{dataset_name}.npy'
    data = np.concatenate((x_train, y_train[:, None]), axis=1)
    np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def road_network_20K_cluster(dataset_dir: Path) -> bool:
    """
    3DRoadNetwork dataset from UCI repository (
    http://archive.ics.uci.edu/ml/datasets/3D+Road+Network+%28North+Jutland%2c+Denmark%29#)
    road_network x cluster dataset (20000, 4)
    """
    dataset_name = 'road_network_20K_cluster'
    os.makedirs(dataset_dir, exist_ok=True)

    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt'

    local_url = os.path.join(dataset_dir, os.path.basename(url))

    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}, train')
        retrieve(url, local_url)
    logging.info(f'{dataset_name} is loaded, started parsing...')

    nrows_train, dtype = 20000, np.float32
    data_train: Any = pd.read_csv(local_url, dtype=dtype,
                                  nrows=nrows_train)

    x_train = np.ascontiguousarray(data_train.values[:nrows_train, 1:], dtype=dtype)
    y_train = np.ascontiguousarray(data_train.values[:nrows_train, 0], dtype=dtype)

    filename = f'{dataset_name}.npy'
    data = np.concatenate((x_train, y_train[:, None]), axis=1)
    np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True
