# Copyright (C) 2017 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import sys
import os
import argparse


anchor_dir = os.path.dirname(__file__)
data_dir = os.path.join(anchor_dir, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

assert len(sys.argv) == 3

X_fn_in = sys.argv[1]
Y_fn_in = sys.argv[2]

assert os.path.exists(X_fn_in)
assert os.path.exists(Y_fn_in)

X = np.load(X_fn_in)
Y = np.load(Y_fn_in)
Y[Y==0] = -1

np.savetxt(X_fn_in + '.csv', X, fmt='%.18e', delimiter=',')
np.savetxt(Y_fn_in + '.csv', Y, fmt='%.18e', delimiter=',')
