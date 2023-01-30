# Anxolotl-engines
## Origin

This work is part of the Anxolotl System and thesis by the name of
Anxolotl - An Anxiety Companion App thesis


# Instructions

## Pre Requisities

- Python 3.8
- An IDE running python (in my case PyCharm)

## Organization

This directory is divided into three sections :

- datasets : Contains information about retrieving the used datasets
- panic_engine : The machine learning algorithm to distinguish between panic states (Non-Panic and panic)
- anxiety_engine : The machine learning algorithm to classify anxiety state from 3 levels

### Anxiety and Panic Engine folder hierarchy

Both of these folders contain identic organizations
- dataset_handlers : Files pertaining the loading and filtering of each specific dataset
- processed_data : Contains a python file with a data processor and a folder **processed_data/data** which contains already processed datasets.
- utils : Graph generation utility code and Logger
- main.py : The entrypoint of the machine learning algorithm
- load_proccess_dataset.py : The entry point to load the original datasets and create filtered datasets compatible with the engines

## Contacts

- Email : [gomes.nunoms@gmail.com](<a href="mailto:gomes.nunoms@gmail.com"></a>)
- Github : [nunogoms](https://github.com/nunogoms)
- Linkedin : [Nuno Gomes](https://www.linkedin.com/in/nuno-g-7734291a5/)

## Citation

If you find this repository useful in your research, please cite this article as

```
@article{
YANG2020295,
title = "On hyperparameter optimization of machine learning algorithms: Theory and practice",
author = "Li Yang and Abdallah Shami",
volume = "415",
pages = "295 - 316",
journal = "Neurocomputing",
year = "2020",
issn = "0925-2312",
doi = "https://doi.org/10.1016/j.neucom.2020.07.061",
url = "http://www.sciencedirect.com/science/article/pii/S0925231220311693"
}
```


