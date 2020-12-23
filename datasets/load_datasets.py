# ===============================================================================
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
# ===============================================================================

import os
import sys
import argparse
import logging

from .loader import (a9a, gisette, ijcnn, skin_segmentation,
                     klaverjas, connect, mnist, sensit, covertype)

dataset_loaders = {
    "a9a": a9a,
    "gisette": gisette,
    "ijcnn": ijcnn,
    "skin_segmentation": skin_segmentation,
    "klaverjas": klaverjas,
    "connect": connect,
    "mnist": mnist,
    "sensit": sensit,
    "covertype": covertype,
}


def try_load_dataset(dataset_name, output_directory):
    if dataset_name in dataset_loaders.keys():
        try:
            return dataset_loaders[dataset_name](output_directory)
        except:
            logging.warning("Internal error loading dataset")
            return False
    else:
        logging.warning(f"There is no script to download the dataset: {dataset_name}. "
                        "You need to add a dataset or script to load it.")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Use \'-d\' or \'--datasets\' option to enumerate '
                    'dataset(s) which should be downloaded')
    parser.add_argument('-l', '--list', action='store_const',
                        const=True, help='list of available datasets')
    parser.add_argument('-d', '--datasets', type=str, nargs='*',
                        help='datasets which should be downloaded')
    args = parser.parse_args()

    if args.list:
        for key in dataset_loaders.keys():
            print(key)
        sys.exit(0)

    root_dir = os.environ['DATASETSROOT']

    if args.datasets is not None:
        for val in dataset_loaders.values():
            val(root_dir)
    elif len(args.datasets) == 0:
        logging.warning(
            'Warning: Enumerate dataset(s) which should be downloaded')
    else:
        for key, val in dataset_loaders.items():
            if key in args.datasets:
                val(root_dir)
