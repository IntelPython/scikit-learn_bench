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

import argparse

from .implementation import DIFFBY_COLUMNS


def add_report_generator_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--report-log-level",
        default="WARNING",
        type=str,
        choices=("ERROR", "WARNING", "INFO", "DEBUG"),
        help="Logging level for report generator.",
    )
    parser.add_argument(
        "--result-files",
        type=str,
        nargs="+",
        default=list(),
        help="Result file path[s] from scikit-learn_bench runs for report generation.",
    )
    parser.add_argument(
        "--report-file", type=str, default="report.xlsx", help="Report file path."
    )
    parser.add_argument(
        "--report-type",
        type=str,
        default="separate-tables",
        choices=("separate-tables",),
        help='Report type ("separate-tables" is the only supported now).',
    )
    parser.add_argument(
        "--compatibility-mode",
        default=False,
        action="store_true",
        help="[EXPERIMENTAL] Compatibility mode drops and modifies results "
        "to make them comparable (for example, sklearn and cuML parameters).",
    )
    # 'separate-table' report type arguments
    parser.add_argument(
        "--drop-columns",
        "--drop-cols",
        type=str,
        nargs="+",
        default=list(),
        help="Columns to drop from report.",
    )
    parser.add_argument(
        "--diff-columns",
        "--diff-cols",
        type=str,
        nargs="+",
        default=DIFFBY_COLUMNS,
        help="Columns to show difference between.",
    )
    parser.add_argument(
        "--split-columns",
        type=str,
        nargs="+",
        default=["estimator", "method", "function"],
        help="Splitting columns for subreports/sheets.",
    )
    parser.add_argument(
        "--diffs-selection",
        type=str,
        choices=["upper_triangle", "lower_triangle", "matrix"],
        default="upper_triangle",
        help="Selects which part of one-vs-one difference to show "
        "(all matrix or one of triangles).",
    )
    # color scale settings
    parser.add_argument(
        "--perf-color-scale",
        type=float,
        nargs="+",
        default=[0.8, 1.0, 10.0],
        help="Color scale for performance metric improvement in report.",
    )
    parser.add_argument(
        "--quality-color-scale",
        type=float,
        nargs="+",
        default=[0.99, 0.995, 1.01],
        help="Color scale for quality metric improvement in report.",
    )
    return parser


def get_report_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sklbench.report",
        description="""
            Scikit-learn_bench report generator
            """,
    )
    add_report_generator_arguments(parser)
    return parser
