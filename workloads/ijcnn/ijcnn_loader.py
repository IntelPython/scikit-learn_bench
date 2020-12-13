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

def ijcnn(root_dir=None):
    """
    Author: Danil Prokhorov.
    libSVM,AAD group
    Cite: Danil Prokhorov. IJCNN 2001 neural network competition.
    Slide presentation in IJCNN'01,
    Ford Research Laboratory, 2001. http://www.geocities.com/ijcnn/nnc_ijcnn01.pdf.

    ijcnn X train dataset (153344, 22)
    ijcnn y train dataset (153344, 1)
    ijcnn X test dataset  (38337,  22)
    ijcnn y train dataset (38337,  1)

    """

    dataset_dir = os.path.join(root_dir, 'workloads', 'ijcnn', 'dataset')

    try:
        os.makedirs(dataset_dir)
    except FileExistsError:
        pass

    filename_ijcnn_x_train = os.path.join(dataset_dir, 'ijcnn_x_train.csv')
    filename_ijcnn_y_train = os.path.join(dataset_dir, 'ijcnn_y_train.csv')
    filename_ijcnn_x_test = os.path.join(dataset_dir, 'ijcnn_x_test.csv')
    filename_ijcnn_y_test = os.path.join(dataset_dir, 'ijcnn_y_test.csv')

    X, y = fetch_openml(name='ijcnn', return_X_y=True, as_frame=False, data_home=dataset_dir)
    X = pd.DataFrame(X.todense())
    y = pd.DataFrame(y)

    y[y == -1] = 0

    print('ijcnn dataset is downloaded')
    print('reading CSV file...')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train.to_csv(os.path.join(dataset_dir, filename_ijcnn_x_train), header=False, index=False)
    print(f'ijcnn X train dataset {X_train.shape} is ready to be used')

    y_train.to_csv(os.path.join(dataset_dir, filename_ijcnn_y_train), header=False, index=False)
    print(f'ijcnn y train dataset {y_train.shape} is ready to be used')

    X_test.to_csv(os.path.join(dataset_dir, filename_ijcnn_x_test), header=False, index=False)
    print(f'ijcnn X test dataset {X_test.shape} is ready to be used')

    y_test.to_csv(os.path.join(dataset_dir, filename_ijcnn_y_test), header=False, index=False)
    print(f'ijcnn y train dataset {y_test.shape} is ready to be used')


if __name__ == '__main__':
    root_dir = os.environ['DATASETSROOT']
    ijcnn(root_dir)
