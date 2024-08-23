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
import json
from typing import Dict, List

import openpyxl as xl
import pandas as pd
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from scipy.stats import gmean

from ..utils.common import custom_format, flatten_dict, flatten_list
from ..utils.logger import logger
from .compatibility import transform_results_to_compatible

METRICS = {
    "lower is better": [
        "time[ms]",
        "iterations",
        # classification
        "logloss",
        # regression
        "RMSE",
        # clustering
        "inertia",
        "Davies-Bouldin score",
        # manifold
        # - TSNE
        "Kullback-Leibler divergence",
    ],
    "higher is better": [
        "throughput[samples/ms]",
        # classification
        "accuracy",
        "balanced accuracy",
        "ROC AUC",
        # regression
        "R2",
        # clustering
        "homogeneity",
        "completeness",
        # search
        "recall@10",
    ],
    "indifferent": [
        # SVM
        "support vectors",
        # PCA
        "average log-likelihood",
        "1st component variance ratio",
        # DBSCAN
        # NB: 'n_clusters' is parameter of KMeans while
        # 'clusters' is number of computer clusters by DBSCAN
        "clusters",
    ],
    "incomparable": ["time std[ms]"],
}
METRIC_NAMES = flatten_list([list(METRICS[key]) for key in METRICS])
PERF_METRICS = ["time[ms]", "throughput[samples/ms]"]

COLUMNS_ORDER = [
    # algorithm
    "stage",
    "task",
    "library",
    "estimator",
    "method",
    "function",
    "online_inference_mode",
    "device",
    "environment_name",
    # data
    "dataset",
    "samples",
    "features",
    "format",
    "dtype",
    "order",
    "n_classes",
    "n_clusters",
    "batch_size",
]

DIFFBY_COLUMNS = ["environment_name", "library", "format", "device"]


def geomean_wrapper(a):
    return gmean(a, nan_policy="omit")


def reorder_columns(input_columns: List, columns_order: List = COLUMNS_ORDER) -> List:
    output_columns = list()
    # 1st step: select existing columns from known ordered columns
    for ordered_column in columns_order:
        if ordered_column in input_columns:
            output_columns.append(ordered_column)
            input_columns.remove(ordered_column)
    # 2nd step: add left input columns
    output_columns += input_columns
    return output_columns


def filter_nan_columns(input_df: pd.DataFrame):
    output_df = input_df.copy()
    non_nan_columns = output_df.columns[output_df.isna().mean(axis=0) < 1]
    output_df = output_df[non_nan_columns]
    return output_df


def split_df_by_columns(
    input_df: pd.DataFrame, columns: List, remove_column: bool = True
) -> Dict[str, pd.DataFrame]:
    split_columns = list(set(columns) & set(input_df.columns))
    split_columns = reorder_columns(split_columns, columns)
    value_counts = input_df.value_counts(split_columns, dropna=False, sort=False)
    output_dfs = {}
    for unique_values in value_counts.index:
        index_mask = [
            input_df[column] == unique_value
            for column, unique_value in zip(value_counts.index.names, unique_values)
            if not pd.isna(unique_value)
        ]
        index_mask = pd.DataFrame(index_mask).all(axis=0)
        subset_name = str(unique_values)[1:-1]
        subset_name = subset_name.replace(", ", "|").replace(",", "").replace("'", "")
        subset_name = subset_name.replace("nan|", "").replace("|nan", "")
        output_dfs[subset_name] = filter_nan_columns(input_df.loc[index_mask])
        if remove_column:
            output_dfs[subset_name] = output_dfs[subset_name].drop(
                columns=set(split_columns) & set(output_dfs[subset_name].columns)
            )
        output_dfs[subset_name] = output_dfs[subset_name][
            reorder_columns(list(output_dfs[subset_name].columns))
        ]
    return output_dfs


def compare_df(input_df, diff_columns, diffs_selection, compared_columns=METRIC_NAMES):
    def select_comparison(i, j, diffs_selection):
        if diffs_selection == "upper_triangle":
            return j > i
        elif diffs_selection == "lower_triangle":
            return i > j
        return i != j

    index_columns = list(
        (set(input_df.columns) - set(diff_columns)) - set(compared_columns)
    )
    df = input_df.set_index(index_columns)
    unique_indices = df.index.unique()
    splitted_dfs = split_df_by_columns(input_df, diff_columns)
    splitted_dfs = {key: df.set_index(index_columns) for key, df in splitted_dfs.items()}

    # drop results with duplicated indices (keep first entry only)
    for key, splitted_df in splitted_dfs.items():
        splitted_dfs[key] = splitted_df[~splitted_df.index.duplicated(keep="first")]

    df = pd.DataFrame(index=unique_indices)
    # original values
    for key, splitted_df in splitted_dfs.items():
        if len(set(splitted_df.columns) - set(compared_columns)) > 0:
            raise ValueError
        for column in splitted_df.columns:
            df[f"{key}\n{column}"] = splitted_df[column]
    # compared values
    for i, (key_ith, df_ith) in enumerate(splitted_dfs.items()):
        for j, (key_jth, df_jth) in enumerate(splitted_dfs.items()):
            if select_comparison(i, j, diffs_selection):
                comparison_name = f"{key_jth} vs {key_ith}"
                for column in df_ith.columns:
                    if column in METRICS["higher is better"]:
                        df[f"{comparison_name}\n{column} relative improvement"] = (
                            df_jth[column] / df_ith[column]
                        )
                    elif column in METRICS["lower is better"]:
                        df[f"{comparison_name}\n{column} relative improvement"] = (
                            df_ith[column] / df_jth[column]
                        )
                    elif column in METRICS["indifferent"]:
                        df[f"{comparison_name}\n{column} is equal"] = (
                            df_ith[column] == df_jth[column]
                        )
    df = df.reset_index()
    # move to multi-index
    df = df[reorder_columns(list(df.columns))]
    df.columns = [
        column if "\n" in column else f"parameter\n{column}" for column in df.columns
    ]
    df.columns = pd.MultiIndex.from_tuples(
        [tuple(column.split("\n")) for column in df.columns]
    )
    return df


def write_df_to_sheet(df, sheet, index=True, header=True):
    for row in dataframe_to_rows(df, index=index, header=header):
        if any(map(lambda x: x is not None, row)):
            sheet.append(row)


def merge_result_files(filenames):
    results = dict()
    for result_name in filenames:
        with open(result_name, "r") as fp:
            result = json.load(fp)
        for key, value in result.items():
            if key in results:
                if isinstance(value, list):
                    results[key] += value
                elif isinstance(value, dict):
                    results[key].update(value)
            else:
                results[key] = value
    return results


def get_result_tables_as_df(
    results,
    diffby_columns=DIFFBY_COLUMNS,
    splitby_columns=["estimator", "method", "function"],
    compatibility_mode=False,
):
    bench_cases = pd.DataFrame(
        [flatten_dict(bench_case) for bench_case in results["bench_cases"]]
    )

    if compatibility_mode:
        bench_cases = transform_results_to_compatible(bench_cases)

    for column in diffby_columns.copy():
        if bench_cases[column].nunique() == 1:
            bench_cases.drop(columns=[column], inplace=True)
            diffby_columns.remove(column)

    return split_df_by_columns(bench_cases, splitby_columns)


def get_summary_from_df(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    metric_columns = list()
    for column in list(df.columns):
        for metric_name in METRIC_NAMES:
            # only relative improvements are included in summary currently
            if len(column) > 1 and column[1] == f"{metric_name} relative improvement":
                metric_columns.append(column)
    summary = df[metric_columns].aggregate(geomean_wrapper, axis=0).to_frame().T
    summary.index = pd.Index([df_name])
    return summary


def get_color_rule(scale):
    red, yellow, green = "F85D5E", "FAF52E", "58C144"
    start_value, mid_value, end_value = scale
    return ColorScaleRule(
        start_type="num",
        start_value=start_value,
        start_color=red,
        mid_type="num",
        mid_value=mid_value,
        mid_color=yellow,
        end_type="num",
        end_value=end_value,
        end_color=green,
    )


def apply_rules_for_sheet(sheet, perf_color_scale, quality_color_scale):
    for column in sheet.iter_cols():
        column_idx = get_column_letter(column[0].column)
        is_rel_impr = any(
            [
                isinstance(cell.value, str) and "relative improvement" in cell.value
                for cell in column
            ]
        )
        is_time = any(
            [
                isinstance(cell.value, str)
                and (any(map(lambda x: x in cell.value, PERF_METRICS)))
                for cell in column
            ]
        )
        if is_rel_impr:
            cell_range = f"${column_idx}1:${column_idx}{len(column)}"
            sheet.conditional_formatting.add(
                cell_range,
                get_color_rule(perf_color_scale if is_time else quality_color_scale),
            )


def write_environment_info(results, workbook):
    env_infos = results["environment"]
    for env_name, env_info in env_infos.items():
        for info_type, info_subclass in env_info.items():
            new_ws = workbook.create_sheet(title=f"{info_type}|{env_name}"[:31])
            for sub_key, sub_info in info_subclass.items():
                if isinstance(sub_info, dict):
                    if all(
                        map(
                            lambda x: not (isinstance(x, list) or isinstance(x, dict)),
                            sub_info.values(),
                        )
                    ):
                        info_df = pd.Series(sub_info).to_frame()
                    else:
                        info_df = pd.DataFrame(sub_info).T
                elif isinstance(sub_info, list):
                    info_df = pd.DataFrame(sub_info)
                else:
                    continue
                write_df_to_sheet(info_df, new_ws)
                new_ws.append([None])


def generate_report(args: argparse.Namespace):
    logger.setLevel(args.report_log_level)
    results = merge_result_files(args.result_files)

    diffby, splitby = args.diff_columns, args.split_columns
    dfs = get_result_tables_as_df(results, diffby, splitby, args.compatibility_mode)

    wb = xl.Workbook()
    summary_dfs = list()
    for df_name, df in dfs.items():
        drop_columns = list(set(df.columns) & set(args.drop_columns))
        df = df.drop(columns=drop_columns)

        ws = wb.create_sheet(title=df_name[:30])
        if len(diffby) > 0:
            current_df = compare_df(df, diffby, args.diffs_selection)
        else:
            current_df = df
        write_df_to_sheet(current_df, ws, index=False)
        apply_rules_for_sheet(ws, args.perf_color_scale, args.quality_color_scale)
        summary_dfs.append(get_summary_from_df(current_df, df_name))
    # write summary to corresponding sheet
    summary_df = pd.concat(summary_dfs, axis=0, join="outer")
    summary_df = summary_df[summary_df.columns.sortlevel(level=0, ascending=False)[0]]
    logger.info(f"{custom_format('Report summary', bcolor='HEADER')}\n{summary_df}")
    if summary_df.size > 0:
        summary_ws = wb.create_sheet(title="Summary", index=0)
        write_df_to_sheet(summary_df, summary_ws)
        apply_rules_for_sheet(summary_ws, args.perf_color_scale, args.quality_color_scale)
    # write environment info
    write_environment_info(results, wb)
    # remove default sheet
    wb.remove(wb["Sheet"])
    wb.save(args.report_file)
    return 0
