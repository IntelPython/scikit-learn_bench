# Report generator

**Scikit-learn_bench** report generator creates high-level report with aggregated stats from provided benchmark results.

Generator will support different types of report, but there is only one supported type currently:

 - `Separate tables`: writes aggregated metrics on summary page and detailed result on separate pages

Raw results are converted into pandas.DataFrame and final report is made by processing this dataframe into separate ones written into Excel tables.

## Arguments
<!-- Note: generate arguments table using runner: `python -m sklbench --describe-parser` -->

| Name                               | Type  | Default value                             | Choices                                        | Description                                                                           |
|:-----------------------------------|:------|:------------------------------------------|:-----------------------------------------------|:--------------------------------------------------------------------------------------|
| `--report-log-level`               | str   | WARNING                                   | ('ERROR', 'WARNING', 'INFO', 'DEBUG')          | Logging level for report generator.                                                   |
| `--result-files`                   | str   | ['result.json']                           |                                                | Result file path[s] from scikit-learn_bench runs for report generation.               |
| `--report-file`                    | str   | report.xlsx                               |                                                | Report file path.                                                                     |
| `--report-type`                    | str   | separate-tables                           | ('separate-tables',)                           | Report type ("separate-tables" is the only supported now).                            |
| `--drop-columns`</br>`--drop-cols` | str   | []                                        |                                                | Columns to drop from report.                                                          |
| `--diff-columns`</br>`--diff-cols` | str   | ['environment_hash', 'library', 'device'] |                                                | Columns to show difference between.                                                   |
| `--split-columns`                  | str   | ['estimator', 'method', 'function']       |                                                | Splitting columns for subreports/sheets.                                              |
| `--diffs-selection`                | str   | upper_triangle                            | ['upper_triangle', 'lower_triangle', 'matrix'] | Selects which part of one-vs-one difference to show (all matrix or one of triangles). |
| `--time-color-scale`               | float | [0.8, 1.0, 10.0]                          |                                                | Time improvement color scale in report.                                               |
| `--metric-color-scale`             | float | [0.99, 0.995, 1.01]                       |                                                | Metric improvement color scale in report.                                             |
