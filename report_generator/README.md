# Report generator for scikit-learn_bench

Report generator produces Excel table file from json benchmark log files.

Run `python report_generator.py --result-files bench_log_1.json,bench_log_2.json [--report-file new_report.xlsx --generation-config default_report_gen_config.json]` to launch report generation.

runner options:
* ``result-files`` : comma-separated benchmark json result file paths
* ``report-file`` : report file path
* ``generation-config`` : generation configuration file path

config parameters:
* ``header`` : Column names in the table header. These parameters are also used to compare reports. If the name is compound, then it should be separated by the '':'' symbol
* ``comparison_method`` : Comparison of two results. The options are: ``1 operation 2`` or ``2 operation 1``
* ``aggregation_metrics`` : Metric applied to columns with comparisons of two reports. For each of these metrics, a separate sheet with a summary is compiled. It is important that the function is in Excel. For example: ``geomean`` or ``average``.
