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

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def skin_segmentation(root_dir=None):
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

    skin_segmentation X train dataset (196045, 3)
    skin_segmentation y train dataset (196045, 1)
    skin_segmentation X test dataset  (49012,  3)
    skin_segmentation y train dataset (49012,  1)

    """

    dataset_dir = os.path.join(root_dir, 'workloads', 'skin_segmentation', 'dataset')

    try:
        os.makedirs(dataset_dir)
    except FileExistsError:
        pass

    filename_skin_segmentation_x_train = os.path.join(dataset_dir, 'skin_segmentation_x_train.csv')
    filename_skin_segmentation_y_train = os.path.join(dataset_dir, 'skin_segmentation_y_train.csv')
    filename_skin_segmentation_x_test = os.path.join(dataset_dir, 'skin_segmentation_x_test.csv')
    filename_skin_segmentation_y_test = os.path.join(dataset_dir, 'skin_segmentation_y_test.csv')

    X, y = fetch_openml(name='skin-segmentation', return_X_y=True, as_frame=True, data_home=dataset_dir)
    print(X.shape, y.shape)
    y = y.astype(int)
    y[y == 2] = 0

    print('skin_segmentation dataset is downloaded')
    print('reading CSV file...')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train.to_csv(os.path.join(dataset_dir, filename_skin_segmentation_x_train), header=False, index=False)
    print(f'skin_segmentation X train dataset {X_train.shape} is ready to be used')

    y_train.to_csv(os.path.join(dataset_dir, filename_skin_segmentation_y_train), header=False, index=False)
    print(f'skin_segmentation y train dataset {y_train.shape} is ready to be used')

    X_test.to_csv(os.path.join(dataset_dir, filename_skin_segmentation_x_test), header=False, index=False)
    print(f'skin_segmentation X test dataset {X_test.shape} is ready to be used')

    y_test.to_csv(os.path.join(dataset_dir, filename_skin_segmentation_y_test), header=False, index=False)
    print(f'skin_segmentation y train dataset {y_test.shape} is ready to be used')


if __name__ == '__main__':
    root_dir = os.environ['DATASETSROOT']
    skin_segmentation(root_dir)
