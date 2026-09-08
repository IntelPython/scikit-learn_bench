# Report Generator

**Scikit-learn_bench** report generator creates a high-level report with aggregated stats from provided benchmark results.

Generator will eventually support different types of reports, but there is only one supported type currently:

 - `Separate tables`: writes aggregated metrics on summary page and detailed result on separate pages

Raw results are converted into a pandas dataframe and the final report is made by processing this dataframe into separate ones written into Excel tables.

## Arguments
<!-- Note: generate arguments table using runner: `python -m sklbench --describe-parser` -->

| Name                                           | Type   | Default value                                       | Choices                                        | Description                                                                                                                      |
|:-----------------------------------------------|:-------|:----------------------------------------------------|:-----------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|
| `--report-log-level`                           | str    | WARNING                                             | ('ERROR', 'WARNING', 'INFO', 'DEBUG')          | Logging level for report generator.                                                                                              |
| `--result-files`                               | str    | ['result.json']                                     |                                                | Result file path[s] from scikit-learn_bench runs for report generation.                                                          |
| `--report-file`                                | str    | report.xlsx                                         |                                                | Report file path.                                                                                                                |
| `--report-type`                                | str    | separate-tables                                     | ('separate-tables',)                           | Report type ("separate-tables" is the only supported now).                                                                       |
| `--compatibility-mode`                         |        | False                                               |                                                | [EXPERIMENTAL] Compatibility mode drops and modifies results to make them comparable (for example, sklearn and cuML parameters). |
| `--performance-stability-metrics`</br>`-psm`   |        | False                                               |                                                | Adds performance stability metrics (`1st run time[ms]`, `1st-mean run ratio`, `median time[ms]`, `time CV`) to the report.       |
| `--combined-results`                           |        | False                                               |                                                | [EXPERIMENTAL] Creates `All cases` and `Summary (for plots)` sheets combining time and speedup of all algorithms with per-dtype/total GEOMEAN rows (use with `--compatibility-mode`). |
| `--draw-plots`                                 |        | False                                               |                                                | [EXPERIMENTAL] Draws Training/Inference speedup bar charts from the combined results (requires `--combined-results`). Footnote hardware parameters are hardcoded; edit them in the code for a production-ready plot.             |
| `--plot-output`                                | str    | `plot.png`                                          |                                                | [EXPERIMENTAL] Output file path for plots (e.g. `plots.png`); if unset, plots are saved to `plot.png`. Only takes effect together with `--draw-plots`.                           |
| `--drop-columns`</br>`--drop-cols`             | str    | []                                                  |                                                | Columns to drop from report.                                                                                                     |
| `--diff-columns`</br>`--diff-cols`             | str    | ['environment_name', 'library', 'format', 'device'] |                                                | Columns to show difference between.                                                                                              |
| `--split-columns`                              | str    | ['estimator', 'method', 'function']                 |                                                | Splitting columns for subreports/sheets.                                                                                         |
| `--diffs-selection`                            | str    | upper_triangle                                      | ['upper_triangle', 'lower_triangle', 'matrix'] | Selects which part of one-vs-one difference to show (all matrix or one of triangles).                                            |
| `--perf-color-scale`                           | float  | [0.8, 1.0, 10.0]                                    |                                                | Color scale for performance metric improvement in report.                                                                        |
| `--quality-color-scale`                        | float  | [0.99, 0.995, 1.01]                                 |                                                | Color scale for quality metric improvement in report.                                                                            |

---
[Documentation tree](../../README.md#-documentation)
