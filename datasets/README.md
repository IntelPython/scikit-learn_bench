# Public dataset include in for scikit-learn_bench

## Download datasets

The download selected public datasets included in the benchmark, please run the following command:

```bash
DATASETSROOT=/path/to/local/data/directory python -m datasets.load_datasets -d <DS_NAME_1> <DS_NAME_2>
```

### Important Notes

1. [Download location](#download-location)
2. [Execution mode](#execution-mode)

#### Download location

The scipt relies on a `DATASETSROOT` environment variable to indicate the local path where 
datasets will be automatically downloaded.

You can set this variable directly via command line when launching the script (see example above). 

Alternatively, you can set this variable in your environment **before** running the script:

```shell
export DATASETSROOT=/path/to/local/data/directory
```

#### Execution Mode

Please run the `load_datasets` script from the **main** root directory of the `scikit-learn_bench` benchmark.

Executing the script directly from the `datasets` folder will not work due to issues with relative imports.

## List of available datasets

To access the list of all the datasets included in the benchmark, please use the `--list` option:

```bash
python -m datasets.load_datasets --list
```

## Collect dataset names used in experiments

It is possible to gather the list of public datasets used in benchmark
experiments.
This list can be later used as input to the `load_datasets` script to download
all the data required to run selected benchmarks.

To collect the names of the dataset included in benchmark configuration file(s), please run:

```shell
python collect_dataset_names.py -f config_1.json config_2.json ...
```

The list of dataset name(s) found will be printed on standard output.

Alternatively, please use the `--output` (`-o`) option to specify the path of the output file
where this list will be printed instead:

```shell
python collect_dataset_names.py -f config_1.json config_2.json ... -o dataset_names.txt
```


