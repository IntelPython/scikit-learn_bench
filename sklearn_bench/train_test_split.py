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

import argparse

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
    description='scikit-learn train_test_split benchmark')
parser.add_argument('--train-size', type=float, default=0.75,
                    help='Size of training subset')
parser.add_argument('--test-size', type=float, default=0.25,
                    help='Size of testing subset')
parser.add_argument('--do-not-shuffle', default=False, action='store_true',
                    help='Do not perform data shuffle before splitting')
parser.add_argument('--include-y', default=False, action='store_true',
                    help='Include label (Y) in splitting')
parser.add_argument('--rng', default=None,
                    choices=('MT19937', 'SFMT19937', 'MT2203', 'R250', 'WH',
                             'MCG31', 'MCG59', 'MRG32K3A', 'PHILOX4X32X10',
                             'NONDETERM', None),
                    help='Random numbers generator for shuffling '
                         '(only for IDP scikit-learn)')
params = bench.parse_args(parser)

# Load generated data
X, y, _, _ = bench.load_data(params)

if params.include_y:
    data_args = (X, y)
else:
    data_args = (X, )

tts_params = {
    'train_size': params.train_size,
    'test_size': params.test_size,
    'shuffle': not params.do_not_shuffle,
    'random_state': params.seed
}

if params.rng is not None:
    tts_params['rng'] = params.rng

time, _ = bench.measure_function_time(train_test_split, *data_args, **tts_params,
                                      params=params)

bench.print_output(library='sklearn', algorithm='train_test_split',
                   stages=['training'], params=params,
                   functions=['train_test_split'], times=[time], accuracies=[None],
                   accuracy_type=None, data=[X], alg_params=tts_params)
