# qdevplot

# Description
The qdevplot library is a collection of plots used in Qdev. 
The idea is to share plot functions inside the group and with a future version of your self. 

# Installation

You can install QDevPlot by:

```bash
$ git clone https://github.com/qdev-dk/QDevPlot.git
$ cd QDevPlot
$ pip install -e .
```

# Usage

## Running the tests

If you have gotten 'qdevplot' from source, you may run the tests locally.

Install `qdevplot` along with its test dependencies into your virtual environment by executing the following in the root folder

```bash
$ pip install .
$ pip install -r test_requirements.txt
```

Then run `pytest` in the `tests` folder.

## Building the documentation

If you have gotten `qdevplot` from source, you may build the docs locally.

Install `qdevplot` along with its documentation dependencies into your virtual environment by executing the following in the root folder

```bash
$ pip install .
$ pip install -r docs_requirements.txt
```

You also need to install `pandoc`. If you are using `conda`, that can be achieved by

```bash
$ conda install pandoc
```
else, see [here](https://pandoc.org/installing.html) for pandoc's installation instructions.

Then run `make html` in the `docs` folder. The next time you build the documentation, remember to run `make clean` before you run `make html`.
