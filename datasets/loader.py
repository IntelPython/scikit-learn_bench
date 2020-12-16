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
import logging

import pandas as pd
import numpy as np

from urllib.request import urlretrieve

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def a9a(dataset_dir=None):
    """
    Author: Ronny Kohavi","Barry Becker
    libSVM","AAD group
    Source: original - Date unknown
    Cite: http://archive.ics.uci.edu/ml/datasets/Adult

    Classification task. n_classes = 2.
    a9a X train dataset (39073, 123)
    a9a y train dataset (39073, 1)
    a9a X test dataset  (9769,  123)
    a9a y train dataset (9769,  1)
    """
    dataset_name = 'a9a'
    os.makedirs(dataset_dir, exist_ok=True)

    X, y = fetch_openml(name='a9a', return_X_y=True, as_frame=False, data_home=dataset_dir)
    X = pd.DataFrame(X.todense())
    y = pd.DataFrame(y)

    y[y == -1] = 0

    logging.info('a9a dataset is downloaded')
    logging.info('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    for data, name in zip((x_train, x_test, y_train, y_test), ('x_train', 'x_test', 'y_train', 'y_test')):
        filename =  f'{dataset_name}_{name}.csv'
        data.to_csv(os.path.join(dataset_dir, filename), header=False, index=False)

    logging.info('dataset a9a ready.')
    return True

# def gisette(root_dir=None):
#     """
#     GISETTE is a handwritten digit recognition problem.
#     The problem is to separate the highly confusable digits '4' and '9'.
#     This dataset is one of five datasets of the NIPS 2003 feature selection challenge.

#     Classification task. n_classes = 2.
#     gisette X train dataset (6000, 5000)
#     gisette y train dataset (6000, 1)
#     gisette X test dataset  (1000, 5000)
#     gisette y train dataset (1000, 1)
#     """
#     os.makedirs(dataset_dir, exist_ok=True)

#     filename_gisette_x_train = 'gisette_x_train.csv')
#     filename_gisette_y_train = 'gisette_y_train.csv')
#     filename_gisette_x_test = 'gisette_x_test.csv')
#     filename_gisette_y_test = 'gisette_y_test.csv')

#     gisette_train_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.data'

#     filename_train_data = 'gisette_train.data')
#     if not os.path.exists(filename_train_data):
#         urlretrieve(gisette_train_data_url, filename_train_data)

#     gisette_train_labels_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.labels'
#     filename_train_labels = 'gisette_train.labels')
#     if not os.path.exists(filename_train_labels):
#         urlretrieve(gisette_train_labels_url, filename_train_labels)

#     gisette_test_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_valid.data'
#     filename_test_data = 'gisette_valid.data')
#     if not os.path.exists(filename_test_data):
#         urlretrieve(gisette_test_data_url, filename_test_data)

#     gisette_test_labels_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/gisette/gisette_valid.labels'
#     filename_test_labels = 'gisette_valid.labels')
#     if not os.path.exists(filename_test_labels):
#         urlretrieve(gisette_test_labels_url, filename_test_labels)

#     logging.info('gisette dataset is downloaded')
#     logging.info('reading CSV file...')

#     num_cols = 5000

#     df_train = pd.read_csv(filename_train_data, header = None)
#     df_labels = pd.read_csv(filename_train_labels, header = None)
#     num_train = 6000
#     x_train = df_train.iloc[:num_train].values
#     x_train = pd.DataFrame(np.array([np.fromstring(elem[0], dtype = int, count = num_cols, sep = ' ') for elem in x_train]))
#     y_train = df_labels.iloc[:num_train].values
#     y_train = pd.DataFrame((y_train > 0).astype(int))

#     num_train = 1000
#     df_test = pd.read_csv(filename_test_data, header = None)
#     df_labels = pd.read_csv(filename_test_labels, header = None)
#     x_test = df_test.iloc[:num_train].values
#     x_test = pd.DataFrame(np.array([np.fromstring(elem[0], dtype = int, count = num_cols, sep = ' ') for elem in x_test]))
#     y_test = df_labels.iloc[:num_train].values
#     y_test = pd.DataFrame((y_test > 0).astype(int))

#     x_train.to_csv(filename_gisette_x_train, header=False, index=False)
#     y_train.to_csv(filename_gisette_y_train, header=False, index=False)
#     x_test.to_csv(filename_gisette_x_test, header=False, index=False)
#     y_test.to_csv(filename_gisette_y_tes, header=False, index=False)
#     logging.info('dataset gisette ready.')
#     return True

def ijcnn(root_dir=None):
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
    ijcnn y train dataset (38337,  1)
    """
    os.makedirs(dataset_dir, exist_ok=True)

    filename_ijcnn_x_train = 'ijcnn_x_train.csv'
    filename_ijcnn_y_train = 'ijcnn_y_train.csv'
    filename_ijcnn_x_test = 'ijcnn_x_test.csv'
    filename_ijcnn_y_test = 'ijcnn_y_test.csv'

    X, y = fetch_openml(name='ijcnn', return_X_y=True, as_frame=False, data_home=dataset_dir)
    X = pd.DataFrame(X.todense())
    y = pd.DataFrame(y)

    y[y == -1] = 0

    logging.info('ijcnn dataset is downloaded')
    logging.info('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_train.to_csv(filename_ijcnn_x_train, header=False, index=False)
    y_train.to_csv(filename_ijcnn_y_train, header=False, index=False)
    x_test.to_csv(filename_ijcnn_x_test, header=False, index=False)
    y_test.to_csv(filename_ijcnn_y_test, header=False, index=False)
    logging.info('dataset ijcnn ready.')
    return True

def skin_segmentation(root_dir=None):
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
    skin_segmentation y train dataset (49012,  1)
    """
    os.makedirs(dataset_dir, exist_ok=True)

    filename_skin_segmentation_x_train = 'skin_segmentation_x_train.csv'
    filename_skin_segmentation_y_train = 'skin_segmentation_y_train.csv'
    filename_skin_segmentation_x_test = 'skin_segmentation_x_test.csv'
    filename_skin_segmentation_y_test = 'skin_segmentation_y_test.csv'

    X, y = fetch_openml(name='skin-segmentation', return_X_y=True, as_frame=True, data_home=dataset_dir)
    logging.info(X.shape, y.shape)
    y = y.astype(int)
    y[y == 2] = 0

    logging.info('skin_segmentation dataset is downloaded')
    logging.info('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_train.to_csv(filename_skin_segmentation_x_train, header=False, index=False)
    y_train.to_csv(filename_skin_segmentation_y_train, header=False, index=False)
    x_test.to_csv(filename_skin_segmentation_x_test, header=False, index=False)
    y_test.to_csv(filename_skin_segmentation_y_test, header=False, index=False)
    logging.info('dataset ijcnn ready.')
    return True


def klaverjas(root_dir=None):
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
    klaverjas y train dataset (49012,  1)
    """
    os.makedirs(dataset_dir, exist_ok=True)

    filename_klaverjas_x_train = 'klaverjas_x_train.csv'
    filename_klaverjas_y_train = 'klaverjas_y_train.csv'
    filename_klaverjas_x_test = 'klaverjas_x_test.csv'
    filename_klaverjas_y_test = 'klaverjas_y_test.csv'

    X, y = fetch_openml(name='Klaverjas2018', return_X_y=True, as_frame=True, data_home=dataset_dir)

    y = y.cat.codes
    logging.info('klaverjas dataset is downloaded')
    logging.info('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)
    x_train.to_csv(filename_klaverjas_x_train, header=False, index=False)
    logging.info(f'klaverjas X train dataset {x_train.shape} is ready to be used')

    y_train.to_csv(filename_klaverjas_y_train, header=False, index=False)
    logging.info(f'klaverjas y train dataset {y_train.shape} is ready to be used')

    x_test.to_csv(filename_klaverjas_x_test, header=False, index=False)
    logging.info(f'klaverjas X test dataset {x_test.shape} is ready to be used')

    y_test.to_csv(filename_klaverjas_y_test, header=False, index=False)
    logging.info(f'klaverjas y train dataset {y_test.shape} is ready to be used')
    logging.info('dataset klaverjas ready.')
    return True

def connect(root_dir=None):
    """
    Source:
    UC Irvine Machine Learning Repository
    http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.htm

    Classification task. n_classes = 3.
    connect X train dataset (196045, 127)
    connect y train dataset (196045, 1)
    connect X test dataset  (49012,  127)
    connect y train dataset (49012,  1)
    """
    os.makedirs(dataset_dir, exist_ok=True)

    filename_connect_x_train = 'connect_x_train.csv'
    filename_connect_y_train = 'connect_y_train.csv'
    filename_connect_x_test = 'connect_x_test.csv'
    filename_connect_y_test = 'connect_y_test.csv'

    X, y = fetch_openml(name='connect-4', return_X_y=True, as_frame=False, data_home=dataset_dir)
    X = pd.DataFrame(X.todense())
    y = pd.DataFrame(y)
    y = y.astype(int)

    logging.info('connect dataset is downloaded')
    logging.info('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    x_train.to_csv(filename_connect_x_train, header=False, index=False)
    y_train.to_csv(filename_connect_y_train, header=False, index=False)
    x_test.to_csv(filename_connect_x_test, header=False, index=False)
    y_test.to_csv(filename_connect_y_test, header=False, index=False)

    logging.info('dataset connect ready.')
    return True

def mnist(root_dir=None):
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
    mnist y train dataset (10000,  1)
    """
    os.makedirs(dataset_dir, exist_ok=True)

    filename_mnist_x_train = 'mnist_x_train.csv'
    filename_mnist_y_train = 'mnist_y_train.csv'
    filename_mnist_x_test = 'mnist_x_test.csv'
    filename_mnist_y_test = 'mnist_y_test.csv'

    X, y = fetch_openml(name='mnist_784', return_X_y=True, as_frame=True, data_home=dataset_dir)
    y = y.astype(int)
    X = X / 255
    logging.info(X.shape, y.shape)

    logging.info('mnist dataset is downloaded')
    logging.info('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=10000, shuffle=False)
    x_train.to_csv(filename_mnist_x_train, header=False, index=False)
    logging.info(f'mnist X train dataset {x_train.shape} is ready to be used')

    y_train.to_csv(filename_mnist_y_train, header=False, index=False)
    logging.info(f'mnist y train dataset {y_train.shape} is ready to be used')

    x_test.to_csv(filename_mnist_x_test, header=False, index=False)
    logging.info(f'mnist X test dataset {x_test.shape} is ready to be used')

    y_test.to_csv(filename_mnist_y_test, header=False, index=False)
    logging.info(f'mnist y train dataset {y_test.shape} is ready to be used')
    logging.info('dataset mnist ready.')
    return True

def sensit(root_dir=None):
    """
    Abstract: Vehicle classification in distributed sensor networks.
    Author: M. Duarte, Y. H. Hu
    Source: [original](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets)

    Classification task. n_classes = 2.
    sensit X train dataset (196045, 3)
    sensit y train dataset (196045, 1)
    sensit X test dataset  (49012,  3)
    sensit y train dataset (49012,  1)
    """
    os.makedirs(dataset_dir, exist_ok=True)

    filename_sensit_x_train = 'sensit_x_train.csv'
    filename_sensit_y_train = 'sensit_y_train.csv'
    filename_sensit_x_test = 'sensit_x_test.csv'
    filename_sensit_y_test = 'sensit_y_test.csv'

    X, y = fetch_openml(name='SensIT-Vehicle-Combined', return_X_y=True, as_frame=False, data_home=dataset_dir)
    X = pd.DataFrame(X.todense())
    y = pd.DataFrame(y)
    y = y.astype(int)

    logging.info(X.shape, y.shape)

    logging.info('sensit dataset is downloaded')
    logging.info('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_train.to_csv(filename_sensit_x_train, header=False, index=False)
    logging.info(f'sensit X train dataset {x_train.shape} is ready to be used')

    y_train.to_csv(filename_sensit_y_train, header=False, index=False)
    logging.info(f'sensit y train dataset {y_train.shape} is ready to be used')

    x_test.to_csv(filename_sensit_x_test, header=False, index=False)
    logging.info(f'sensit X test dataset {x_test.shape} is ready to be used')

    y_test.to_csv(filename_sensit_y_test, header=False, index=False)
    logging.info(f'sensit y train dataset {y_test.shape} is ready to be used')
    logging.info('dataset sensit ready.')
    return True


def covertype(root_dir=None):
    """

    covertype X train dataset (196045, 3)
    covertype y train dataset (196045, 1)
    covertype X test dataset  (49012,  3)
    covertype y train dataset (49012,  1)

    """
    os.makedirs(dataset_dir, exist_ok=True)

    filename_covertype_x_train = os.path.join(dataset_dir, 'covertype_x_train.csv')
    filename_covertype_y_train = os.path.join(dataset_dir, 'covertype_y_train.csv')
    filename_covertype_x_test = os.path.join(dataset_dir, 'covertype_x_test.csv')
    filename_covertype_y_test = os.path.join(dataset_dir, 'covertype_y_test.csv')

    X, y = fetch_openml(name='covertype', return_X_y=True, as_frame=True, data_home=dataset_dir)
    y = y.astype(int)

    print(X.shape, y.shape)

    print('covertype dataset is downloaded')
    print('reading CSV file...')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train.to_csv(os.path.join(dataset_dir, filename_covertype_x_train), header=False, index=False)
    print(f'covertype X train dataset {X_train.shape} is ready to be used')

    y_train.to_csv(os.path.join(dataset_dir, filename_covertype_y_train), header=False, index=False)
    print(f'covertype y train dataset {y_train.shape} is ready to be used')

    X_test.to_csv(os.path.join(dataset_dir, filename_covertype_x_test), header=False, index=False)
    print(f'covertype X test dataset {X_test.shape} is ready to be used')

    y_test.to_csv(os.path.join(dataset_dir, filename_covertype_y_test), header=False, index=False)
    print(f'covertype y train dataset {y_test.shape} is ready to be used')