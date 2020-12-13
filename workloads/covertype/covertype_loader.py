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

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def covertype(root_dir=None):
    """
    Abstract:
    The Skin Segmentation dataset is constructed over B, G, R color space.
    Skin and Nonskin dataset is generated using skin textures from
    face images of diversity of age, gender, and race people.

    Source:
    Rajen Bhatt, Abhinav Dhall, rajen.bhatt '@' gmail.com, IIT Delhi.

    Data Set Information:
    The skin dataset is collected by randomly sampling B,G,R values from face
    images of various age groups (young, middle, and old),
    race groups (white, black, and asian), and genders obtained from
    FERET database and PAL database.
    Total learning sample size is 245057; out of which 50859
    is the skin samples and 194198 is non-skin samples.

    Attribute Information:
    This dataset is of the dimension 245057 * 4
    where first three columns are B,G,R (x1,x2, and x3 features)
    values and fourth column is of the class labels (decision variable y).

    covertype X train dataset (196045, 3)
    covertype y train dataset (196045, 1)
    covertype X test dataset  (49012,  3)
    covertype y train dataset (49012,  1)

    """

    dataset_dir = os.path.join(root_dir, 'workloads', 'covertype', 'dataset')

    try:
        os.makedirs(dataset_dir)
    except FileExistsError:
        pass

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


if __name__ == '__main__':
    root_dir = os.environ['DATASETSROOT']
    covertype(root_dir)
