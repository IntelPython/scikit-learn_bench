# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import division, print_function

import itertools
import numpy as np
import os
from sklearn.datasets import make_classification

def gen_datasets(features, vectors, classes, dest='data', prefix=""):
    """Generate classification datasets in binary .npy files

    features: a list of feature lengths to test
    vectors: a list of sample lengths to test
    classes: number of classes (2 for binary classification dataset)
    """
    for f, v in itertools.product(features, vectors):
        X, y = make_classification(n_samples=v, n_features=f, n_informative=f, n_redundant=0, n_classes=classes, random_state=0)
        np.save(os.path.join(dest, prefix + "X-{}x{}.npy".format(v, f)), X)
        np.save(os.path.join(dest, prefix + "y-{}x{}.npy".format(v, f)), y)
