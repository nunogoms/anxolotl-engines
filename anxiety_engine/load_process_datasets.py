import os

import numpy as np
import typing

import sklearn.preprocessing

from dataset_handlers.anxiety_phases_dataset import AnxietyPhasesDatasetOper
from dataset_handlers.common_dataset_entry import CommonDatasetEntry
from dataset_handlers.mmash_dataset import MmashDatasetOper
from processed_data.hrv_analysis_oper import HrvAnalysisOper
from dotenv import load_dotenv

from utils import paths

######      LOADING ENVIRONMENT CONFIGS      ######
load_dotenv(dotenv_path=f"{paths.get_project_root()}/{paths.LOADER_CONFIG_PATH}")

# Train models information

train_model_filename_env = os.environ.get("TRAINED_MODEL_FILENAME")
train_model_storage_path = os.environ.get("TRAINED_MODELS_STORAGE_PATH")

# Features array information

output_features_array_path = os.environ.get("OUTPUT_FEATURES_ARRAY_PATH")

MMASH_DATASET_PATH: str = os.environ.get("MMASH_DATASET_PATH")
ANXIETY_PHASES_DATASET_PATH: str = os.environ.get("ANXIETY_PHASES_DATASET")

time_between_samples_in_seconds: int =  int(os.environ.get("TIME_BETWEEN_SAMPLES_IN_SECONDS_TO_RESAMPLE"))
completeness_percentage_for_measures: float = float(os.environ.get("COMPLETENESS_PERCENTAGE_OF_MEASURES_PER_HOUR"))

mmash_dataset = MmashDatasetOper(MMASH_DATASET_PATH)
anxiety_phases_dataset = AnxietyPhasesDatasetOper(ANXIETY_PHASES_DATASET_PATH)


features_list = []
labels_list = []

common_dataset_entry_list :typing.List[CommonDatasetEntry]= []
common_dataset_entry_list.extend(anxiety_phases_dataset.retrieve_features_and_labels(reduce_labels=True, selected_task=0, time_between_samples=time_between_samples_in_seconds))
common_dataset_entry_list.extend(anxiety_phases_dataset.retrieve_features_and_labels(reduce_labels=True, selected_task=1, time_between_samples=time_between_samples_in_seconds))
common_dataset_entry_list.extend(mmash_dataset.retrieve_features_and_labels(reduce_labels=True, last_user=23, time_between_samples = time_between_samples_in_seconds))

for dataset_entry in common_dataset_entry_list :
    resultsArr: typing.List = [
        dataset_entry.user_gender,  # gender
        dataset_entry.user_age,
        np.mean(dataset_entry.rr),
        np.var(dataset_entry.rr),
        np.percentile(dataset_entry.rr, q=90),
        np.percentile(dataset_entry.rr, q=10),
        np.percentile(dataset_entry.rr, q=80) - np.percentile(
            dataset_entry.rr, q=20)
    ]
    resultsArr.extend(HrvAnalysisOper.get_max_hour(dataset_entry.rr,
                                                   dataset_entry.timeline,
                                                   time_between_samples= time_between_samples_in_seconds,
                                                   completeness_percentage=completeness_percentage_for_measures))


    features_list.append(resultsArr)
    labels_list.append(dataset_entry.anxiety_label)

#scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = sklearn.preprocessing.StandardScaler()
features_list = scaler.fit_transform(features_list)

#Save files
np.savetxt(f"{output_features_array_path}/dataset_{time_between_samples_in_seconds}sec_labels.csv", labels_list, delimiter=",")
np.savetxt(f"{output_features_array_path}/dataset_{time_between_samples_in_seconds}sec_features.csv", features_list, delimiter=",")