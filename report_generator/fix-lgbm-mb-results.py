# ==============================================================================
# Copyright 2020-2023 Intel Corporation
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
# ==============================================================================

"""
Temporary solution to fix the .json result files created from lgbm_mb.py.
The result files are in an incompatible format for report_generator.py.
Attempts to produce xlsx reports fail and create empty files.

After running this script on my-file.json, a new file my-file-fixed.json will be
produced, containing a JSON version of the results in a compatible format.

Usage:

  python fix-lgbm-mb-results.py my-file.json [another-file.json ...]


Note: This is just a quick and dirty hack that does not fix the underlying
      issue. Rather than changing this file (if something breaks again), the
      original script lgbm_mb.py should be updated such that it produces valid
      JSON dumps again.
"""

from argparse import ArgumentParser
import json
from pathlib import Path


def fix_file(fname: Path):
    with open(fname) as fp:
        data = json.load(fp)

    # copy all data (aux info etc)
    fixed = {}
    for key, val in data.items():
        fixed[key] = val

    # reset the results - we'll fix them
    fixed["results"] = []

    current_result = {}
    for result in data["results"]:
        if "algorithm" in result:
            # found a new algo / measurement
            current_result = result
            continue

        if "stage" in result:
            comb = current_result | result
            if "device" not in comb:
                comb["device"] = "none"

            if "time[s]" not in comb:
                comb["time[s]"] = (
                    result.get("training_time") or result["prediction_time"]
                )

            if "algorithm_parameters" not in comb:
                comb["algorithm_paramters"] = {}

            if "accuracy[%]" in comb:
                comb["accuracy"] = comb["accuracy[%]"]

            replace_pairs = (
                ("lgbm_train", "training"),
                ("lgbm_predict", "prediction"),
                ("daal4py_predict", "alternative_prediction"),
            )
            for s, r in replace_pairs:
                comb["stage"] = comb["stage"].replace(s, r)

            fixed["results"].append(comb)

    out_fname = fname.stem + "-fixed.json"
    with open(out_fname, "w") as fp:
        json.dump(fixed, fp, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filenames", nargs="+")
    args = parser.parse_args()
    for fname in args.filenames:
        fix_file(Path(fname))
