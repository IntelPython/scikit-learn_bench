import argparse
import json

from ..utils.logger import logger


def main_template(main_method):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-case", required=True, type=str)
    parser.add_argument("--filters", required=True, type=str)
    parser.add_argument(
        "--log-level",
        default="WARNING",
        type=str,
        choices=("ERROR", "WARNING", "INFO", "DEBUG"),
        help="Logging level for benchmark",
    )
    args = parser.parse_args()

    logger.setLevel(args.log_level)

    bench_case = json.loads(args.bench_case)
    filters = json.loads(args.filters)["filters"]

    results = main_method(bench_case, filters)
    print(json.dumps(results, indent=4))
