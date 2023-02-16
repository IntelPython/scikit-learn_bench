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
import json
import sys

def extract_public_dataset_names(exp_filepath: str) -> set[str]:
    with open(exp_filepath) as json_config_file:
        experiment = json.load(json_config_file)
        if not "cases" in experiment:
            return []
        dataset_names = list()
        for case in experiment["cases"]:
            if "dataset" not in case:
                continue
            for ds in case["dataset"]:
                if ds["source"] == "synthethic" or "name" not in ds:
                    continue
                dataset_names.append(ds["name"])
    return set(dataset_names)  # remove duplicates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility to gather the list of public dataset names included in benchmark configuration file(s)."
    )
    parser.add_argument(
        "-f",
        "--files",
        type=str,
        nargs="*",
        help="Benchmark configuration file(s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Writes collected dataset names into an output file. One name per line.",
        default=""
    )
    args = parser.parse_args()

    if not args.files:
        print("Error: Missing input benchmark configuration file(s) to analyze.")
        sys.exit(-1)

    names = set()
    for config_file in args.files:
        names = names.union(extract_public_dataset_names(config_file))

    if len(names):
        print(f"Found {len(names)} dataset(s)")
        if args.output:
            with open(args.output, "w") as output_file:
                for name in names:
                    output_file.write(f"{name}\n")
            print(f"Saved in {args.output}")
        else:
            for name in names:
                print(f"{name}")
    else:
        logging.error("Warning: No public dataset found in input benchmark file(s).")