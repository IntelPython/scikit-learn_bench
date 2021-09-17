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
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.preprocessing import StandardScaler

from .loader_utils import retrieve


def abalone(dataset_dir: Path) -> bool:
    """
    https://archive.ics.uci.edu/ml/machine-learning-databases/abalone

    abalone x train dataset (3341, 8)
    abalone y train dataset (3341, 1)
    abalone x test dataset  (836,  8)
    abalone y train dataset (836,  1)
    """
    dataset_name = 'abalone'
    os.makedirs(dataset_dir, exist_ok=True)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
    local_url = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        retrieve(url, local_url)
    logging.info(f'{dataset_name} is loaded, started parsing...')

    abalone: Any = pd.read_csv(local_url, header=None)
    abalone[0] = abalone[0].astype('category').cat.codes
    X = abalone.iloc[:, :-1].values
    y = abalone.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)

    for data, name in zip((X_train, X_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def california_housing(dataset_dir: Path) -> bool:
    """
    california_housing x train dataset (18576, 8)
    california_housing y train dataset (18576, 1)
    california_housing x test dataset  (2064,  8)
    california_housing y train dataset (2064,  1)
    """
    dataset_name = 'california_housing'
    os.makedirs(dataset_dir, exist_ok=True)

    X, y = fetch_california_housing(return_X_y=True, as_frame=False,
                                    data_home=dataset_dir)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    logging.info(f'{dataset_name} is loaded, started parsing...')

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    scaler = StandardScaler().fit(x_train, y_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    for data, name in zip((x_train, x_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def fried(dataset_dir: Path) -> bool:
    """
    fried x train dataset (32614, 10)
    fried y train dataset (32614, 1)
    fried x test dataset  (8154,  10)
    fried y train dataset (8154,  1)
    """
    dataset_name = 'fried'
    os.makedirs(dataset_dir, exist_ok=True)

    X, y = fetch_openml(
        name='fried', return_X_y=True, as_frame=False, data_home=dataset_dir)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    logging.info(f'{dataset_name} is loaded, started parsing...')

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    for data, name in zip((x_train, x_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def medical_charges_nominal(dataset_dir: Path) -> bool:
    """
    medical_charges_nominal x train dataset (130452, 11)
    medical_charges_nominal y train dataset (130452, 1)
    medical_charges_nominal x test dataset  (32613,  11)
    medical_charges_nominal y train dataset (32613,  1)
    """
    dataset_name = 'medical_charges_nominal'
    os.makedirs(dataset_dir, exist_ok=True)

    X, y = fetch_openml(name='medical_charges_nominal', return_X_y=True,
                        as_frame=False, data_home=dataset_dir)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    logging.info(f'{dataset_name} is loaded, started parsing...')

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(x_train, y_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    scaler = StandardScaler().fit(y_train)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)

    for data, name in zip((x_train, x_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def mortgage_first_q(dataset_dir: Path) -> bool:
    """
    # TODO: add an loading instruction
    """
    return False


def twodplanes(dataset_dir: Path) -> bool:
    """
    twodplanes x train dataset (106288, 10)
    twodplanes y train dataset (106288, 1)
    twodplanes x test dataset  (70859,  10)
    twodplanes y train dataset (70859,  1)
    """
    dataset_name = 'twodplanes'
    os.makedirs(dataset_dir, exist_ok=True)

    X, y = fetch_openml(
        name='BNG(2dplanes)', return_X_y=True, as_frame=False, data_home=dataset_dir)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    logging.info(f'{dataset_name} is loaded, started parsing...')

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42)
    for data, name in zip((x_train, x_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def year_prediction_msd(dataset_dir: Path) -> bool:
    """
    YearPredictionMSD dataset from UCI repository
    https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd

    year_prediction_msd x train dataset (463715, 90)
    year_prediction_msd y train dataset (463715, 1)
    year_prediction_msd x test dataset  (51630,  90)
    year_prediction_msd y train dataset (51630,  1)
    """
    dataset_name = 'year_prediction_msd'
    os.makedirs(dataset_dir, exist_ok=True)

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/' \
          'YearPredictionMSD.txt.zip'
    local_url = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        retrieve(url, local_url)
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


def yolanda(dataset_dir: Path) -> bool:
    """
    yolanda x train dataset (130452, 11)
    yolanda y train dataset (130452, 1)
    yolanda x test dataset  (32613,  11)
    yolanda y train dataset (32613,  1)
    """
    dataset_name = 'yolanda'
    os.makedirs(dataset_dir, exist_ok=True)

    X, y = fetch_openml(name='yolanda', return_X_y=True,
                        as_frame=False, data_home=dataset_dir)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    logging.info(f'{dataset_name} is loaded, started parsing...')

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(x_train, y_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    for data, name in zip((x_train, x_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True


def airline_regression(dataset_dir: Path) -> bool:
    """
    yolanda x train dataset (8500000, 9)
    yolanda y train dataset (8500000, 1)
    yolanda x test dataset  (1500000, 9)
    yolanda y train dataset (1500000, 1)
    """
    dataset_name = 'airline_regression'
    os.makedirs(dataset_dir, exist_ok=True)

    X, y = fetch_openml(name='Airlines_DepDelay_10M', return_X_y=True,
                        as_frame=False, data_home=dataset_dir)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    logging.info(f'{dataset_name} is loaded, started parsing...')

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(x_train, y_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    scaler = StandardScaler().fit(y_train)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)

    for data, name in zip((x_train, x_test, y_train, y_test),
                          ('x_train', 'x_test', 'y_train', 'y_test')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True
