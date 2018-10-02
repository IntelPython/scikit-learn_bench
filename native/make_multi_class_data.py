# Copyright (C) 2017 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import sys
import os
import argparse

assert len(sys.argv) == 3

X_fn_in = sys.argv[1]
Y_fn_in = sys.argv[2]

assert os.path.exists(X_fn_in)
assert os.path.exists(Y_fn_in)

X = np.load(X_fn_in)
Y = np.load(Y_fn_in)
assert np.min(Y) >= 0
assert np.allclose(np.unique(Y), np.arange(0, np.max(Y)+1))

np.savetxt(X_fn_in + '.csv', X, fmt='%.18e', delimiter=',')
np.savetxt(Y_fn_in + '.csv', Y, fmt='%.18e', delimiter=',')
