# ===============================================================================
# Copyright 2024 Intel Corporation
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

import sys

from sklbench.runner import get_parser_description, get_runner_parser, run_benchmarks


def main():
    parser = get_runner_parser()
    args = parser.parse_args()
    if args.describe_parser:
        print(get_parser_description(parser))
        return 0
    else:
        return run_benchmarks(args)


if __name__ == "__main__":
    sys.exit(main())
