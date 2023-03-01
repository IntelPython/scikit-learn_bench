# ===============================================================================
# Copyright 2020-2021 Intel Corporation
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
# ===============================================================================

import json

import numpy as np


def get_accuracy(true_labels, prediction):
    errors = 0
    for i, true_label in enumerate(true_labels):
        pred_label = 0
        if isinstance(prediction[i], (np.float32, np.float64)):
            pred_label = prediction[i] > 0.5
        elif prediction[i].shape[0] == 1:
            pred_label = prediction[i][0]
        else:
            pred_label = np.argmax(prediction[i])
        if true_label != pred_label:
            errors += 1
    return 100 * (1 - errors / len(true_labels))


def print_output(library, algorithm, stages, params, functions,
                 times, metric_type, metrics, data):
    if params.output_format == 'json':
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
        for i, stage in enumerate(stages):
            result = {
                'stage': stage,
            }
            if 'daal' in stage:
                result.update({'conversion_to_daal4py': times[2 * i],
                               'prediction_time': times[2 * i + 1]})
            elif 'train' in stage:
                result.update({'matrix_creation_time': times[2 * i],
                               'training_time': times[2 * i + 1]})
            else:
                result.update({'matrix_creation_time': times[2 * i],
                               'prediction_time': times[2 * i + 1]})
            if metrics[i] is not None:
                result.update({f'{metric_type}': metrics[i]})
            output.append(result)
        print(json.dumps(output, indent=4))
