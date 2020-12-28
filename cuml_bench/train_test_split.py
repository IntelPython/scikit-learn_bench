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

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench
from cuml import train_test_split

parser = argparse.ArgumentParser(
    description='cuml train_test_split benchmark')
parser.add_argument('--train-size', type=float, default=0.75,
                    help='Size of training subset')
parser.add_argument('--test-size', type=float, default=0.25,
                    help='Size of testing subset')
parser.add_argument('--do-not-shuffle', default=False, action='store_true',
                    help='Do not perform data shuffle before splitting')
params = bench.parse_args(parser)

# Load generated data
X, y, _, _ = bench.load_data(params)

tts_params = {
    'train_size': params.train_size,
    'test_size': params.test_size,
    'shuffle': not params.do_not_shuffle,
    'random_state': params.seed
}

time, _ = bench.measure_function_time(train_test_split, X=X, y=y, params=params)

bench.print_output(library='cuml', algorithm='train_test_split',
                   stages=['training'], params=params,
                   functions=['train_test_split'], times=[time], accuracies=[None],
                   accuracy_type=None, data=[X], alg_params=tts_params)
