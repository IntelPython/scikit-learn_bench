# Report generator for scikit-learn_bench

Report generator produces Excel table file from json benchmark log files.

Run `python report_generator.py --result-files bench_log_1.json,bench_log_2.json [--report-file new_report.xlsx --generation-config default_report_gen_config.json]` to launch report generation.

runner options:
* ``result-files`` : comma-separated benchmark json result file paths
* ``report-file`` : report file path
* ``generation-config`` : generation configuration file path

config parameters:
* ``header``: The column names in the table header. These parameters are also used to compare reports. If a name is compound, use the ``:`` symbol to separate its parts.
* ``comparison_method``: The formula for the comparison of two results. The options are: ``1 operation 2`` or ``2 operation 1``, where ``1`` is the first result and ``2`` is the second result. The default is ``2 / 1``, which returns the ratio of the second result to the first one.
* ``aggregation_metrics``: The metrics applied to the columns with the comparisons of two reports. You can use multiple metrics. For each of these metrics, a separate sheet with a summary is compiled. The metrics should be Excel functions. For example: ``"geomean", "average"``.
