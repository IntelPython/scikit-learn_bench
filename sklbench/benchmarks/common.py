# ===============================================================================
# Copyright 2023 Intel Corporation
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
import json

from ..utils.logger import logger


def main_template(main_method):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-case", required=True, type=str)
    parser.add_argument("--filters", required=True, type=str)
    parser.add_argument(
        "--log-level",
        default="WARNING",
        type=str,
        choices=("ERROR", "WARNING", "INFO", "DEBUG"),
        help="Logging level for benchmark",
    )
    args = parser.parse_args()

    logger.setLevel(args.log_level)

    bench_case = json.loads(args.bench_case)
    filters = json.loads(args.filters)["filters"]

    results = main_method(bench_case, filters)
    print(json.dumps(results, indent=4))
