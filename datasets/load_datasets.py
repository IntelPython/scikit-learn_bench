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

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict

from .loader_classification import (a_nine_a, airline, airline_ohe, bosch,
                                    census, codrnanorm, creditcard, epsilon, fraud,
                                    gisette, higgs, higgs_one_m, ijcnn,
                                    klaverjas, santander, skin_segmentation)
from .loader_multiclass import (connect, covertype, covtype, letters, mlsr,
                                mnist, msrank, plasticc, sensit)
from .loader_regression import (abalone, california_housing, fried,
                                medical_charges_nominal, mortgage_first_q,
                                twodplanes, year_prediction_msd, yolanda)

dataset_loaders: Dict[str, Callable[[Path], bool]] = {
    "a9a": a_nine_a,
    "abalone": abalone,
    "airline": airline,
    "airline-ohe": airline_ohe,
    "bosch": bosch,
    "california_housing": california_housing,
    "census": census,
    "codrnanorm": codrnanorm,
    "connect": connect,
    "covertype": covertype,
    "covtype": covtype,
    "creditcard": creditcard,
    "epsilon": epsilon,
    "fraud": fraud,
    "fried": fried,
    "gisette": gisette,
    "higgs": higgs,
    "higgs1m": higgs_one_m,
    "ijcnn": ijcnn,
    "klaverjas": klaverjas,
    "letters": letters,
    "mlsr": mlsr,
    "medical_charges_nominal": medical_charges_nominal,
    "mnist": mnist,
    "mortgage1Q": mortgage_first_q,
    "msrank": msrank,
    "plasticc": plasticc,
    "santander": santander,
    "sensit": sensit,
    "skin_segmentation": skin_segmentation,
    "twodplanes": twodplanes,
    "year_prediction_msd": year_prediction_msd,
    "yolanda": yolanda,
}


def try_load_dataset(dataset_name: str, output_directory: Path) -> bool:
    if dataset_name in dataset_loaders:
        try:
            return dataset_loaders[dataset_name](output_directory)
        except BaseException as ex:
            logging.warning(f"Internal error loading dataset:\n{ex}")
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
        for key in dataset_loaders:
            print(key)
        sys.exit(0)

    root_dir = Path(os.environ['DATASETSROOT'])

    if args.datasets is not None:
        for val in dataset_loaders.values():
            val(root_dir)
    else:
        logging.warning(
            'Warning: Enumerate dataset(s) which should be downloaded')
