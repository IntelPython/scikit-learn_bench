# Copyright (C) 2017-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT


from bench import print_header, print_row
import json
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


def print_output(library, algorithm, stages, columns, params, functions,
                 times, accuracy_type, accuracies, data):
    if params.output_format == 'csv':
        print_header(columns, params)
        for i in range(len(accuracies)):
            print_row(
                columns, params, prep_function=functions[2 * i],
                function=functions[2 * i + 1],
                time=times[2 * i], prep_time=times[2 * i + 1],
                accuracy=accuracies[i])
    elif params.output_format == 'json':
        output = []
        output.append({
            'library': library,
            'algorithm': algorithm,
            'input_data': {
                'data_format': params.data_format,
                'data_order': params.data_order,
                'data_type': str(params.dtype),
                'dataset_name': params.dataset_name,
                'rows': data[0].shape[0],
                'columns': data[0].shape[1]
            }
        })
        if hasattr(params, 'n_classes'):
            output[-1]['input_data'].update({'classes': params.n_classes})
        for i in range(len(stages)):
            result = {
                'stage': stages[i],
            }
            if 'daal' in stages[i]:
                result.update({'conversion_to_daal4py': times[2 * i],
                               'prediction_time': times[2 * i + 1]})
            elif 'train' in stages[i]:
                result.update({'matrix_creation_time': times[2 * i],
                               'training_time': times[2 * i + 1]})
            else:
                result.update({'matrix_creation_time': times[2 * i],
                               'prediction_time': times[2 * i + 1]})
            if accuracies[i] is not None:
                result.update({f'{accuracy_type}': accuracies[i]})
            output.append(result)
        print(json.dumps(output, indent=4))
