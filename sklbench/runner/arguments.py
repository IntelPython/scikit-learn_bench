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
from typing import Dict, List

import pandas as pd

from ..report import add_report_generator_arguments


def get_parser_description(parser: argparse.ArgumentParser) -> pd.DataFrame:
    """Convert parser description to Markdown-style table."""

    def get_argument_actions(parser: argparse.ArgumentParser) -> List:
        arg_actions = []

        for action in parser._actions:
            if isinstance(action, argparse._ArgumentGroup):
                for subaction in action._group_actions:
                    arg_actions.append(subaction)
            else:
                arg_actions.append(action)
        return arg_actions

    def parse_action(action: argparse.Action) -> Dict:
        return {
            "Name": "</br>".join(map(lambda x: f"`{x}`", action.option_strings)),
            "Type": action.type.__name__ if action.type is not None else None,
            "Default value": (
                action.default if action.default is not argparse.SUPPRESS else None
            ),
            "Choices": action.choices,
            "Description": action.help,
        }

    return pd.DataFrame(map(parse_action, get_argument_actions(parser))).to_markdown(
        index=False
    )


def add_runner_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # verbosity levels
    parser.add_argument(
        "--runner-log-level",
        default="WARNING",
        type=str,
        choices=("ERROR", "WARNING", "INFO", "DEBUG"),
        help="Logging level for benchmarks runner.",
    )
    parser.add_argument(
        "--bench-log-level",
        default="WARNING",
        type=str,
        choices=("ERROR", "WARNING", "INFO", "DEBUG"),
        help="Logging level for each running benchmark.",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        default=None,
        type=str,
        choices=("ERROR", "WARNING", "INFO", "DEBUG"),
        help="Global logging level for benchmarks: "
        "overwrites runner, benchmarks and report logging levels.",
    )
    # benchmarking cases finding, overwriting and filtering
    parser.add_argument(
        "--config",
        "--configs",
        "-c",
        type=str,
        nargs="+",
        default=None,
        help="Paths to a configuration files or/and "
        "directories that contain configuration files.",
    )
    parser.add_argument(
        "--parameters",
        "--params",
        "-p",
        default="",
        type=str,
        nargs="+",
        help="Globally defines or overwrites config parameters. "
        "For example: `-p data:dtype=float32 data:order=F`.",
    )
    parser.add_argument(
        "--parameter-filters",
        "--filters",
        "-f",
        default="",
        type=str,
        nargs="+",
        help="Filters benchmarking cases by parameter values. "
        "For example: `-f data:dtype=float32 data:order=F`.",
    )

    parser.add_argument(
        "--result-file",
        "-r",
        type=str,
        default="result.json",
        help="File path to store scikit-learn_bench's runned cases results.",
    )
    parser.add_argument(
        "--environment-name",
        "--env-name",
        "-e",
        type=str,
        default=None,
        help="Environment name to use instead of it's configuration hash.",
    )
    parser.add_argument(
        "--prefetch-datasets",
        default=False,
        action="store_true",
        help="Load all requested datasets in parallel before running benchmarks.",
    )
    # workflow control
    parser.add_argument(
        "--exit-on-error",
        default=False,
        action="store_true",
        help="Interrupt runner and exit if last benchmark failed with error.",
    )
    # option to get parser description in Markdown table format for READMEs
    parser.add_argument(
        "--describe-parser",
        default=False,
        action="store_true",
        help="Print parser description in Markdown table format and exit.",
    )
    # report generator arguments for optional usage
    parser.add_argument(
        "--report",
        default=False,
        action="store_true",
        help="Enables generation of report.",
    )
    add_report_generator_arguments(parser)
    return parser


def get_runner_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sklbench",
        description="""
            Scikit-learn_bench runner
            """,
    )
    add_runner_arguments(parser)
    return parser
