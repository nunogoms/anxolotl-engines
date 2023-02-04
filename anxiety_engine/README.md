# Anxiety Engine

## Organization

This engine is divided in multiple folders. Their content is:
- [configs](configs) : Contains *.env files with the changeable parameters to run the scripts.
- [dataset_handlers](dataset_handlers) : Files pertaining the loading and filtering of each specific dataset
- [processed_data](processed_data) : Contains a python file with a data processor and a folder **processed_data/data** which contains already processed datasets.
- [utils](utils) : Graph generation utility code and Logger
- main.py : The entry point of the machine learning algorithm
- load_process_dataset.py : The entry point to load the original datasets and create filtered datasets compatible with the engines


## Instructions

This engine contains some scripts to ease the task of setting up and running the engines.
The engine itself have two runnable scripts, one focused on loading and processing the
datasets (load_process_datasets.py) and another one focused on running the machine
learning pipeline and assessing anxiety levels (main.py).

### Scripts

These scripts are able to be run in Linux, Windows(WSL) and Mac.

1. *init.sh* : Used to setup the pyenv and pipenv tools to be able to run the scripts.
  _SHOULD_ be run first.
2. *load_process_datasets.py* : The script containing the code to load and process both
   datasets. Contains the entry parameters at configs/load_config.env.
3. *main.py* : The script containing the code to use the processed datasets and analyse
   for anxiety levels. Contains the entry parameters at configs/main_config.env.

### Commands

i. To configure the environment.


```bash
pyenv init
eval "$(pyenv init -)"
pyenv exec python -m venv .venv
source .venv/bin/activate
pipenv shell
#Write 'exit' on pipenv shell
pipenv lock
pipenv sync
```

ii. To run the script to run and process the dataset.
```bash
source .venv/bin/activate
python load_process_datasets.py 
```

iii. To run the main script to evaluate the anxiety levels.
```bash
source .venv/bin/activate
python main.py 
```