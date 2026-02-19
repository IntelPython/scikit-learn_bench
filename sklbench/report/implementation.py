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

import numpy as np
import openpyxl as xl
import pandas as pd
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from scipy.stats import gmean

from ..utils.common import custom_format, flatten_list
from ..utils.logger import logger
from ..utils.measurement import enrich_metrics
from .compatibility import transform_results_to_compatible

METRICS = {
    "lower is better": [
        "1st run time[ms]",
        "time[ms]",
        "cost[microdollar]",
        "iterations",
        # classification
        "logloss",
        # regression
        "RMSE",
        # clustering
        "inertia",
        "Davies-Bouldin score",
        # manifold - TSNE
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
    "incomparable": [
        "1st-mean run ratio",
        "time CV",
        "cpu load[%]",
    ],
}
MEMORY_TYPES = ["RAM", "VRAM"]
for memory_type in MEMORY_TYPES:
    METRICS["incomparable"].append(f"peak {memory_type} usage[MB]")
    METRICS["incomparable"].append(f"{memory_type} usage-iteration correlation")
METRIC_NAMES = flatten_list([list(METRICS[key]) for key in METRICS])
PERF_METRICS = ["time[ms]", "throughput[samples/ms]", "cost[microdollar]"]

COLUMNS_ORDER = [
    # algorithm
    "stage",
    "task",
    "library",
    "estimator",
    "method",
    "function",
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

RED_COLOR, YELLOW_COLOR, GREEN_COLOR, WHITE_COLOR = "F85D5E", "FAF52E", "58C144", "FFFFFF"
COLUMN_COLOR_RULES = {
    "time CV": ColorScaleRule(
        start_type="num",
        start_value=0.0,
        start_color=GREEN_COLOR,
        mid_type="num",
        mid_value=0.1,
        mid_color=YELLOW_COLOR,
        end_type="num",
        end_value=0.5,
        end_color=RED_COLOR,
    )
}

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
    for key, df in splitted_dfs.items():
        for index_column in index_columns:
            if index_column not in df.columns:
                df[index_column] = np.nan
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
                    if column not in df_jth.columns:
                        continue
                    if column in METRICS["higher is better"]:
                        df[f"{comparison_name}\n{column} relative improvement"] = (
                            df_jth[column] / df_ith[column]
                        )
                    elif column in METRICS["lower is better"]:
                        df[f"{comparison_name}\n{column} relative improvement"] = (
                            df_ith[column] / df_jth[column]
                        )
                    elif column in METRICS["indifferent"]:
                        df[f"{comparison_name}\n{column} is equal"] = df_ith[column].eq(
                            df_jth[column]
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
    include_performance_stability_metrics=False,
):
    bench_cases = pd.DataFrame(
        [
            enrich_metrics(bench_case, include_performance_stability_metrics)
            for bench_case in results["bench_cases"]
        ]
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


def get_color_rule_for_comparison(scale):
    start_value, mid_value, end_value = scale
    return ColorScaleRule(
        start_type="num",
        start_value=start_value,
        start_color=RED_COLOR,
        mid_type="num",
        mid_value=mid_value,
        mid_color=WHITE_COLOR,
        end_type="num",
        end_value=end_value,
        end_color=GREEN_COLOR,
    )


def apply_rules_for_sheet(sheet, perf_color_scale, quality_color_scale):
    for column in sheet.iter_cols():
        column_idx = get_column_letter(column[0].column)
        cell_range = f"${column_idx}1:${column_idx}{len(column)}"
        is_rel_impr = any(
            [
                isinstance(cell.value, str) and "relative improvement" in cell.value
                for cell in column
            ]
        )
        is_perf = any(
            [
                isinstance(cell.value, str)
                and (any(map(lambda x: x in cell.value, PERF_METRICS)))
                for cell in column
            ]
        )
        if is_rel_impr:
            sheet.conditional_formatting.add(
                cell_range,
                get_color_rule_for_comparison(
                    perf_color_scale if is_perf else quality_color_scale
                ),
            )
        else:
            column_name = {cell.value for cell in column} & set(COLUMN_COLOR_RULES.keys())
            if len(column_name) == 1:
                column_name = column_name.pop()
                sheet.conditional_formatting.add(
                    cell_range, COLUMN_COLOR_RULES[column_name]
                )


def write_all_cases_sheet_with_groups(all_cases_df: pd.DataFrame, sheet, perf_color_scale, quality_color_scale):
    """
    Write all cases data to sheet with algorithm groups separated by borders
    and individual color scales per group on comparison columns only.
    Uses green-yellow-red color scale with values computed per group.
    """
    thick_border_top = Border(
        top=Side(style='thick'),

    )
    thick_border_bottom = Border(
        bottom=Side(style='thick')
    )
    
    # Get algorithm name column
    algo_col_name = None
    for col in all_cases_df.columns:
        if isinstance(col, tuple) and col[0] == "algorithm" and col[1] == "name":
            algo_col_name = col
            break
    
    if algo_col_name is None:
        # Fallback: just write normally
        write_df_to_sheet(all_cases_df, sheet, index=False)
        return
    
    # Write header
    header_row = list(all_cases_df.columns)
    header_row_str = ["|".join(col) if isinstance(col, tuple) else str(col) for col in header_row]
    sheet.append(header_row_str)
    
    # Group data by algorithm name
    grouped = all_cases_df.groupby(algo_col_name)
    current_row = 2
    
    for algo_idx, (algo_name, group_df) in enumerate(grouped):
        group_start_row = current_row
        
        # Write group data
        for _, row in group_df.iterrows():
            row_data = [row[col] for col in all_cases_df.columns]
            sheet.append(row_data)
            current_row += 1
        
        group_end_row = current_row - 1
        
        # Apply borders to group (thick on top and bottom)
        for row_num in range(group_start_row, group_end_row + 1):
            for col_idx in range(1, len(all_cases_df.columns) + 1):
                cell = sheet.cell(row=row_num, column=col_idx)
                if row_num == group_start_row:
                    cell.border = thick_border_top
                elif row_num == group_end_row:
                    cell.border = thick_border_bottom
        
        # Apply color scales per group only on comparison columns
        for col_idx, col_name in enumerate(all_cases_df.columns, 1):
            col_letter = get_column_letter(col_idx)
            group_range = f"${col_letter}${group_start_row}:${col_letter}${group_end_row}"
            
            col_str = "|".join(col_name) if isinstance(col_name, tuple) else str(col_name)
            
            # Check if this is a comparison column (contains "vs" or "relative improvement")
            is_comparison = "vs" in col_str or "relative improvement" in col_str
            
            if is_comparison:
                # Get min and max values for this column in this group
                group_values = group_df[col_name].dropna()
                
                if len(group_values) > 0:
                    min_val = group_values.min()
                    max_val = group_values.max()
                    mid_val = (min_val + max_val) / 2
                    
                    # Create red-yellow-green color scale (red for lowest, green for highest)
                    color_rule = ColorScaleRule(
                        start_type="num",
                        start_value=min_val,
                        start_color=RED_COLOR,  # Red for lowest values
                        mid_type="num",
                        mid_value=mid_val,
                        mid_color=YELLOW_COLOR,  # Yellow for middle
                        end_type="num",
                        end_value=max_val,
                        end_color=GREEN_COLOR,  # Green for highest values
                    )
                    sheet.conditional_formatting.add(group_range, color_rule)



def prepare_all_cases_df(all_cases_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare All cases dataframe with specific column ordering:
    1. Algorithm name (df_name from multi-index)
    2. Parameters and other columns
    3. time[ms] related columns
    4. Exclude metric columns (except time[ms])
    """
    df = all_cases_df.copy()
    
    # Flatten multi-index columns for easier processing
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["|".join(col).strip() for col in df.columns.values]
    
    # Identify column groups
    algorithm_cols = [col for col in df.columns if col.startswith("parameter|")]
    time_cols = [col for col in df.columns if "time[ms]" in col and "parameter" not in col]
    
    # Get all metric columns to exclude (except time[ms])
    metric_cols_to_exclude = [
        col for col in df.columns
        if any(metric in col for metric in METRIC_NAMES)
        and "time[ms]" not in col
        and "parameter" not in col
    ]
    
    # Get remaining columns (parameters) - exclude metrics
    remaining_cols = [
        col for col in df.columns
        if col not in algorithm_cols
        and col not in time_cols
        and col not in metric_cols_to_exclude
    ]
    
    ordered_cols = remaining_cols + time_cols + algorithm_cols 
    
    # Filter to only include columns that exist
    ordered_cols = [col for col in ordered_cols if col in df.columns]
    # Select only the ordered columns
    df = df[ordered_cols]
    
    # Convert back to multi-index columns if there were multi-index columns
    if "|" in ordered_cols[0] if ordered_cols else False:
        df.columns = pd.MultiIndex.from_tuples(
            [tuple(col.split("|")) for col in df.columns]
        )
    
    return df


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
    dfs = get_result_tables_as_df(
        results,
        diffby,
        splitby,
        args.compatibility_mode,
        args.performance_stability_metrics,
    )

    wb = xl.Workbook()
    summary_dfs = list()
    all_dfs = list()
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
        # Add algorithm name column for tracking in all_cases sheet
        current_df_with_name = current_df.copy()
        current_df_with_name.insert(0, ("algorithm", "name"), df_name)
        all_dfs.append(current_df_with_name)
    # write summary to corresponding sheet
    all_cases_df = pd.concat(all_dfs, axis=0, join="outer")
    summary_df = pd.concat(summary_dfs, axis=0, join="outer")
    summary_df = summary_df[summary_df.columns.sortlevel(level=0, ascending=False)[0]]
    logger.info(f"{custom_format('Report summary', bcolor='HEADER')}\n{summary_df}")
    if summary_df.size > 0:
        summary_ws = wb.create_sheet(title="Summary", index=0)
        write_df_to_sheet(summary_df, summary_ws)
        apply_rules_for_sheet(summary_ws, args.perf_color_scale, args.quality_color_scale)
    if (all_cases_df.size > 0) and args.combined_results:
        # Prepare all_cases_df with proper column ordering
        all_cases_df = prepare_all_cases_df(all_cases_df)
        all_cases_ws = wb.create_sheet(title="All cases", index=1)
        write_all_cases_sheet_with_groups(all_cases_df, all_cases_ws, args.perf_color_scale, args.quality_color_scale)
    # write environment info
    write_environment_info(results, wb)
    # remove default sheet
    wb.remove(wb["Sheet"])
    wb.save(args.report_file)
    return 0
