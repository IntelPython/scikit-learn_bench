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

from .loader_utils import retrieve


def abalone(dataset_dir: Path) -> bool:
    """
    https://archive.ics.uci.edu/ml/machine-learning-databases/abalone

    TaskType:regression
    NumberOfFeatures:8
    NumberOfInstances:4177
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


def mortgage_first_q(dataset_dir: Path) -> bool:
    """
    # TODO: add an loading instruction
    """
    return False


def year_prediction_msd(dataset_dir: Path) -> bool:
    """
    YearPredictionMSD dataset from UCI repository
    https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd

    TaskType:regression
    NumberOfFeatures:90
    NumberOfInstances:515345
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
