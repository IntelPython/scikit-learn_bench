# Report generator for scikit-learn_bench

Report generator produces Excel table file from json benchmark log files.

Run `python report_generator.py --result-files bench_log_1.json,bench_log_2.json [--report-file new_report.xlsx --generation-config gen_config.json --merging none]` to launch report generation.

runner options:
* ``result-files`` : comma-separated benchmark json result file paths
* ``report-file`` : report file path
* ``generation-config`` : generation configuration file path
* ``merging``: *full*, *none*, *sw_only*, *hw_only*. How to merge same cases in benchmark logs
