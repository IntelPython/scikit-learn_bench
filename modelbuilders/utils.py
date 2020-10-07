# Copyright (C) 2017-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT


import numpy as np


def get_accuracy(true_labels, prediction):
    errors = 0
    for i in range(len(true_labels)):
        pred_label = 0
        if isinstance(prediction[i], float) or \
                isinstance(prediction[i], np.single) or \
                isinstance(prediction[i], np.float):
            pred_label = prediction[i] > 0.5
        elif prediction[i].shape[0] == 1:
            pred_label = prediction[i][0]
        else:
            pred_label = np.argmax(prediction[i])
        if true_labels[i] != pred_label:
            errors += 1
    return 100 * (1 - errors/len(true_labels))
