import argparse
import os
import subprocess
import json


parser = argparse.ArgumentParser()
parser.add_argument("--config", metavar="ConfigPath", type=str,
                    default="config.json")
parser.add_argument("--output", metavar="OutputPath", type=str,
                    default="stdout")
args = parser.parse_args()

with open(args.config, "r") as config_file:
    config = json.load(config_file)


# generate benchmarking cases
def gen_cases(params):
    global cases
    if len(params) == 0:
        return cases
    prev_lenght = len(cases)
    param_name = list(params.keys())[0]
    n_param_values = len(params[param_name])
    cases = cases * n_param_values
    for i in range(n_param_values):
        for j in range(prev_lenght):
            cases[prev_lenght * i + j] += " --{} {}".format(
                param_name, params[param_name][i])
    del params[param_name]
    gen_cases(params)


log = ""
stderr_file = open("_stderr.log", "w")
for algorithm in config["algorithms"]:
    print(algorithm)
    cases = [""]
    params = config["algorithms"][algorithm].copy()
    del params["dataset"]
    gen_cases(params)
    for dataset in config["algorithms"][algorithm]["dataset"]:
        if dataset.startswith("synth"):
            if dataset.startswith("synth_reg"):
                _, _, samples, features = dataset.split("_")
                xfile = "data/reg/X-{}x{}.npy".format(samples, features)
                yfile = "data/reg/y-{}x{}.npy".format(samples, features)
                paths = "-x {} -y {}".format(xfile, yfile)
                command = "python make_datasets.py -s {} -f {} regression -x {} -y {}".format(
                    samples, features, xfile, yfile)
                print(command)
                os.system(command)
            elif dataset.startswith("synth_cls"):
                _, _, classes, samples, features = dataset.split("_")
                xfile = "data/{}/X-{}x{}.npy".format(
                    "two" if classes == 2 else "multi", samples, features)
                yfile = "data/{}/y-{}x{}.npy".format(
                    "two" if classes == 2 else "multi", samples, features)
                paths = "-x {} -y {}".format(xfile, yfile)
                command = "python make_datasets.py -s {} -f {} classification -c {} -x {} -y {}".format(
                    samples, features, classes, xfile, yfile)
                print(command)
                os.system(command)
            else:
                raise ValueError("Unknown dataset type")
        else:
            raise ValueError(
                "Unknown dataset. Only synthetics are supported now")

        for lib in config["libs"]:
            for i, case in enumerate(cases):
                command = "python {}/{}.py --output-format json{} {}".format(
                    lib, algorithm, case, paths)
                print(command)
                r = subprocess.run(command.split(" "), stdout=subprocess.PIPE,
                                   stderr=stderr_file, encoding="utf-8")
                log += r.stdout

while "}\n{" in log:
    log = log.replace("}\n{", "},\n{")

if args.output == "stdout":
    print(log, end="")
else:
    with open(args.output, "w") as output:
        output.write(log)
stderr_file.close()
