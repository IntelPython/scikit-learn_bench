# Download Datasets for scikit-learn_bench

The download selected public datasets included in the benchmark, please run the following command:

```bash
DATASETSROOT=/path/to/local/download/directory python -m datasets.load_datasets -d <DS_NAME_1> <DS_NAME_2>
```

The scipt relies on a `DATASETSROOT` environment variable, to indicate the local path where
datasets will be automatically downloaded.

You can alternatively export this variable in your SHELL environment **before** running the script:

```shell
export DATASETSROOT=/path/to/download/directory
```
## Important Note

Please **do not** run the `load_datasets` script from within the `datasets` folder. This will not work
due to issues with relative imports. 

Please execute the `load_datasets` script directly from the _main_ folder, using the [`-m`](https://docs.python.org/3/using/cmdline.html#cmdoption-m) option with the Python interpreter.


## List of available datasets

To access the list of all the datasets included in the benchmark, please use the `--list` option:

```bash
python -m datasets.load_datasets --list
```

## Download datasets included in configurations files

It is also possible to gather the list of dataset(s) to download directly from
benchmark configuration files by using the `--configs` (`-c`) option:

```shell
DATASETSROOT=/path/to/download/dir python -m datasets.load_datasets -c config_1.json config_2.json ...
```

This method will override the `-d` option, and it is highly recommended when
running multiple benchmark experiments.