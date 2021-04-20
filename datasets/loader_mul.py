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
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.model_selection import train_test_split

from .loader_utils import count_lines, read_libsvm_msrank, retrieve


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
    """
    Cover type dataset from UCI machine learning repository
    https://archive.ics.uci.edu/ml/datasets/covertype

    y contains 7 unique class labels from 1 to 7 inclusive.
    TaskType:multiclass
    NumberOfFeatures:54
    NumberOfInstances:581012
    """
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


def letters(dataset_dir: Path) -> bool:
    """
    http://archive.ics.uci.edu/ml/datasets/Letter+Recognition

    TaskType:multiclass
    NumberOfFeatures:16
    NumberOfInstances:20.000
    """
    dataset_name = 'letters'
    os.makedirs(dataset_dir, exist_ok=True)

    url = ('http://archive.ics.uci.edu/ml/machine-learning-databases/' +
           'letter-recognition/letter-recognition.data')
    local_url = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        retrieve(url, local_url)
    logging.info(f'{dataset_name} is loaded, started parsing...')

    letters = pd.read_csv(local_url, header=None)
    X = letters.iloc[:, 1:].values
    y: Any = letters.iloc[:, 0]
    y = y.astype('category').cat.codes.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    for data, name in zip((X_train, X_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
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
    unzipped_url = os.path.join(dataset_dir, "MSRank")
    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        retrieve(url, local_url)
    if not os.path.isdir(unzipped_url):
        logging.info(f'{dataset_name} is loaded, unzipping...')
        tar = tarfile.open(local_url, "r:gz")
        tar.extractall(dataset_dir)
        tar.close()
    logging.info(f'{dataset_name} is unzipped, started parsing...')

    sets = []
    labels = []
    n_features = 137

    for set_name in ['train.txt', 'vali.txt', 'test.txt']:
        file_name = os.path.join(unzipped_url, set_name)

        n_samples = count_lines(file_name)
        with open(file_name, 'r') as file_obj:
            X, y = read_libsvm_msrank(file_obj, n_samples, n_features, np.float32)

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
    """
    # TODO: add an loading instruction
    """
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
