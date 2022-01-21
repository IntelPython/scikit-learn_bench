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
                                    census, codrnanorm, cifar_binary, creditcard, epsilon,
                                    epsilon_30K, fraud, gisette, hepmass_150K, higgs,
                                    higgs_one_m, ijcnn, klaverjas, santander, skin_segmentation,
                                    susy)
from .loader_multiclass import (connect, covertype, covtype, letters, mlsr,
                                mnist, msrank, plasticc, sensit)
from .loader_regression import (abalone, california_housing, fried, higgs_10500K,
                                medical_charges_nominal, mortgage_first_q,
                                twodplanes, year_prediction_msd, yolanda, airline_regression)
from .loader_clustering import (cifar_cluster, epsilon_50K_cluster, higgs_one_m_clustering,
                                hepmass_1M_cluster, hepmass_10K_cluster, mnist_10K_cluster,
                                road_network_20K_cluster, susy_cluster)

dataset_loaders: Dict[str, Callable[[Path], bool]] = {
    "a9a": a_nine_a,
    "abalone": abalone,
    "airline": airline,
    "airline-ohe": airline_ohe,
    "airline_regression": airline_regression,
    "bosch": bosch,
    "california_housing": california_housing,
    "census": census,
    "cifar_binary": cifar_binary,
    "cifar_cluster": cifar_cluster,
    "codrnanorm": codrnanorm,
    "connect": connect,
    "covertype": covertype,
    "covtype": covtype,
    "creditcard": creditcard,
    "epsilon": epsilon,
    "epsilon_30K": epsilon_30K,
    "epsilon_50K_cluster": epsilon_50K_cluster,
    "fraud": fraud,
    "fried": fried,
    "gisette": gisette,
    "hepmass_150K": hepmass_150K,
    "hepmass_1M_cluster": hepmass_1M_cluster,
    "hepmass_10K_cluster": hepmass_10K_cluster,
    "higgs": higgs,
    "higgs1m": higgs_one_m,
    "higgs_10500K": higgs_10500K,
    "higgs_one_m_clustering": higgs_one_m_clustering,
    "ijcnn": ijcnn,
    "klaverjas": klaverjas,
    "letters": letters,
    "mlsr": mlsr,
    "medical_charges_nominal": medical_charges_nominal,
    "mnist": mnist,
    "mnist_10K_cluster": mnist_10K_cluster,
    "mortgage1Q": mortgage_first_q,
    "msrank": msrank,
    "plasticc": plasticc,
    "road_network_20K_cluster": road_network_20K_cluster,
    "santander": santander,
    "sensit": sensit,
    "skin_segmentation": skin_segmentation,
    "susy_cluster": susy_cluster,
    "susy": susy,
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
                    'dataset(s) that should be downloaded')
    parser.add_argument('-l', '--list', action='store_const',
                        const=True, help='The list of available datasets')
    parser.add_argument('-d', '--datasets', type=str, nargs='*',
                        help='The datasets that should be downloaded.')
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
            'Warning: Enumerate dataset(s) that should be downloaded')
