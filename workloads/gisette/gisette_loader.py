#===============================================================================
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
#===============================================================================

import os
import sys
import pandas as pd
import numpy as np
from urllib.request import urlretrieve

def gisette(root_dir=None):
    """
    GISETTE is a handwritten digit recognition problem.
    The problem is to separate the highly confusable digits '4' and '9'.
    This dataset is one of five datasets of the NIPS 2003 feature selection challenge.

    The digits have been size-normalized and centered in a fixed-size image of dimension 28x28.
    The original data were modified for the purpose of the feature selection challenge.
    In particular, pixels were samples at random in the middle top part of
    the feature containing the information necessary to disambiguate 4 from 9
    and higher order features were created as products of these pixels
    to plunge the problem in a higher dimensional feature space.
    We also added a number of distractor features called 'probes' having no predictive power.
    The order of the features and patterns were randomized.

    Preprocessing: The data set is also available at UCI.
    Because the labels of testing set are not available,
    here we use the validation set (gisette_valid.data and gisette_valid.labels)
    as the testing set. The training data (gisette_train)
    are feature-wisely scaled to [-1,1].
    Then the testing data (gisette_valid) are scaled based on the same scaling factors for the training data.

    gisette X train dataset (6000, 5000)
    gisette y train dataset (6000, 1)
    gisette X test dataset  (1000, 5000)
    gisette y train dataset (1000, 1)

    """

    dataset_dir = os.path.join(root_dir, 'workloads', 'gisette', 'dataset')

    try:
        os.makedirs(dataset_dir)
    except FileExistsError:
        pass

    filename_gisette_X_train = os.path.join(dataset_dir, 'gisette_x_train.csv')
    filename_gisette_y_train = os.path.join(dataset_dir, 'gisette_y_train.csv')
    filename_gisette_X_test = os.path.join(dataset_dir, 'gisette_x_test.csv')
    filename_gisette_y_test = os.path.join(dataset_dir, 'gisette_y_test.csv')

    gisette_train_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.data'

    filename_train_data = os.path.join(dataset_dir, 'gisette_train.data')
    if not os.path.exists(filename_train_data):
        urlretrieve(gisette_train_data_url, filename_train_data)

    gisette_train_labels_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.labels'
    filename_train_labels = os.path.join(dataset_dir, 'gisette_train.labels')
    if not os.path.exists(filename_train_labels):
        urlretrieve(gisette_train_labels_url, filename_train_labels)

    gisette_test_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_valid.data'
    filename_test_data = os.path.join(dataset_dir, 'gisette_valid.data')
    if not os.path.exists(filename_test_data):
        urlretrieve(gisette_test_data_url, filename_test_data)

    gisette_test_labels_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/gisette_valid.labels'
    filename_test_labels = os.path.join(dataset_dir, 'gisette_valid.labels')
    if not os.path.exists(filename_test_labels):
        urlretrieve(gisette_test_labels_url, filename_test_labels)

    print('gisette dataset is downloaded')
    print('reading CSV file...')

    num_cols = 5000

    df_train = pd.read_csv(filename_train_data, header = None)
    df_labels = pd.read_csv(filename_train_labels, header = None)
    num_train = 6000
    X_train = df_train.iloc[:num_train].values
    X_train = pd.DataFrame(np.array([np.fromstring(elem[0], dtype = int, count = num_cols, sep = ' ') for elem in X_train]))
    y_train = df_labels.iloc[:num_train].values
    y_train = pd.DataFrame((y_train > 0).astype(int))

    num_train = 1000
    df_test = pd.read_csv(filename_test_data, header = None)
    df_labels = pd.read_csv(filename_test_labels, header = None)
    X_test = df_test.iloc[:num_train].values
    X_test = pd.DataFrame(np.array([np.fromstring(elem[0], dtype = int, count = num_cols, sep = ' ') for elem in X_test]))
    y_test = df_labels.iloc[:num_train].values
    y_test = pd.DataFrame((y_test > 0).astype(int))

    X_train.to_csv(os.path.join(dataset_dir, filename_gisette_X_train), header=False, index=False)
    print(f'gisette X train dataset {X_train.shape} is ready to be used')

    y_train.to_csv(os.path.join(dataset_dir, filename_gisette_y_train), header=False, index=False)
    print(f'gisette y train dataset {y_train.shape} is ready to be used')

    X_test.to_csv(os.path.join(dataset_dir, filename_gisette_X_test), header=False, index=False)
    print(f'gisette X test dataset {X_test.shape} is ready to be used')

    y_test.to_csv(os.path.join(dataset_dir, filename_gisette_y_test), header=False, index=False)
    print(f'gisette y train dataset {y_test.shape} is ready to be used')


if __name__ == '__main__':
    root_dir = os.environ['DATASETSROOT']
    gisette(root_dir)
