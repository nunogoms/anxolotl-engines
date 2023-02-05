import os
from os import cpu_count

import numpy as np
from typing import List

from sklearn.preprocessing import StandardScaler

from dataset_handlers.anxiety_phases_dataset import AnxietyPhasesDatasetOper
from dataset_handlers.common_dataset_entry import CommonDatasetEntry
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

ANXIETY_PHASES_DATASET_PATH: str = os.environ.get("ANXIETY_PHASES_DATASET")
ANXIETY_PHASES_DATASET_MINI: bool = os.environ.get("ANXIETY_PHASES_DATASET_MINI").lower() == "true"

time_between_samples_in_seconds: int =  int(os.environ.get("TIME_BETWEEN_SAMPLES_IN_SECONDS_TO_RESAMPLE"))

anxiety_phases_dataset = AnxietyPhasesDatasetOper(ANXIETY_PHASES_DATASET_PATH)


features_list = []
labels_list = []

common_dataset_entry_list :List[CommonDatasetEntry]= []
common_dataset_entry_list.extend(anxiety_phases_dataset.retrieve_features_and_labels(is_mini_dataset=ANXIETY_PHASES_DATASET_MINI, selected_task=0, time_between_samples=time_between_samples_in_seconds))
common_dataset_entry_list.extend(anxiety_phases_dataset.retrieve_features_and_labels(is_mini_dataset=ANXIETY_PHASES_DATASET_MINI, selected_task=1, time_between_samples=time_between_samples_in_seconds))

for dataset_entry in common_dataset_entry_list :
    results_arr: List = [
        dataset_entry.user_gender,  # gender
        dataset_entry.user_age,
        np.mean(dataset_entry.rr),
        np.var(dataset_entry.rr),
        np.percentile(dataset_entry.rr, q=90),
        np.percentile(dataset_entry.rr, q=10),
        np.percentile(dataset_entry.rr, q=80) - np.percentile(
            dataset_entry.rr, q=20)
    ]
    #resultsArr.extend(HrvAnalysisOper.getMaxHour(dataset_entry.rr,dataset_entry.timeline,False))
    results_arr.extend(HrvAnalysisOper.get_hrv_features(dataset_entry.rr))

    #resultsArr.extend(HrvAnalysisOper.getHrvAnalysisFeatures(dataset_entry.rr))

    features_list.append(results_arr)
    labels_list.append(dataset_entry.anxiety_label)

#scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = StandardScaler()
features_list = scaler.fit_transform(features_list)

#Save files
np.savetxt(f"{output_features_array_path}/panic_dataset_{time_between_samples_in_seconds}sec_labels.csv", labels_list, delimiter=",")
np.savetxt(f"{output_features_array_path}/panic_dataset_{time_between_samples_in_seconds}sec_features.csv", features_list, delimiter=",")