import os
from os import cpu_count

import numpy as np
from typing import List

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dataset_handlers.anxiety_phases_dataset import AnxietyPhasesDatasetOper
from dataset_handlers.common_dataset_entry import CommonDatasetEntry
from dataset_handlers.mmash_dataset import MmashDatasetOper
from processed_data.hrv_analysis_oper import HrvAnalysisOper
from dotenv import load_dotenv

from utils import paths

######      LOADING ENVIRONMENT CONFIGS      ######
load_dotenv(dotenv_path=f"{paths.get_project_root()}/{paths.MAIN_CONFIG_PATH}")

test = os.environ.get("INPUT_FEATURES_VALUES_NAME")

# Train models information

save_train_model_flag_env = os.environ.get("SAVE_TRAINED_MODELS_FLAG").lower() == "true"
train_model_filename_env = os.environ.get("TRAINED_MODEL_FILENAME")
train_model_storage_path = os.environ.get("TRAINED_MODELS_STORAGE_PATH")

# Features array information

input_features_array_path = os.environ.get("INPUT_FEATURES_ARRAY_PATH")
input_features_values_filename = os.environ.get("INPUT_FEATURES_VALUES_FILENAME")
input_features_labels_filename = os.environ.get("INPUT_FEATURES_LABELS_FILENAME")

MMASH_DATASET_PATH: str = "../datasets/MMASH"
ANXIETY_PHASES_DATASET_PATH: str = "../datasets/AnxietyPhasesDataset"

mmash_dataset_load = MmashDatasetOper(MMASH_DATASET_PATH)
anxiety_phases_dataset_oper = AnxietyPhasesDatasetOper(ANXIETY_PHASES_DATASET_PATH)


features_list = []
labels_list = []

time_between_samples_in_seconds = 30

common_dataset_entry_list :List[CommonDatasetEntry]= []
common_dataset_entry_list.extend(anxiety_phases_dataset_oper.retrieve_features_and_labels(reduce_labels=True,selected_task=0,time_between_samples=time_between_samples_in_seconds))
common_dataset_entry_list.extend(anxiety_phases_dataset_oper.retrieve_features_and_labels(reduce_labels=True,selected_task=1,time_between_samples=time_between_samples_in_seconds))
common_dataset_entry_list.extend(mmash_dataset_load.retrieve_features_and_labels(reduce_labels=True,last_user=23, time_between_samples = time_between_samples_in_seconds))

for dataset_entry in common_dataset_entry_list :
    resultsArr: List = [
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
                                                   completeness_percentage=0.65))


    features_list.append(resultsArr)
    labels_list.append(dataset_entry.anxiety_label)

#scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = StandardScaler()
features_list = scaler.fit_transform(features_list)

#Save files
np.savetxt("processed_data/allsets_" + time_between_samples_in_seconds.__str__() +"sec_final_labels.csv", labels_list, delimiter=",")
np.savetxt("processed_data/allsets_" + time_between_samples_in_seconds.__str__() +"sec_final_features.csv", features_list, delimiter=",")