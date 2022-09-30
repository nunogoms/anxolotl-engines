from os import cpu_count

import numpy as np
from typing import List

from sklearn.preprocessing import StandardScaler

from dataset_handlers.anxiety_phases_dataset import AnxietyPhasesDatasetOper
from dataset_handlers.common_dataset_entry import CommonDatasetEntry
from processed_data.hrv_analysis_oper import HrvAnalysisOper

ANXTIETY_PHASES_DATASET_PATH: str = "../datasets/AnxietyPhasesDataset"

anxiety_phases_dataset_oper = AnxietyPhasesDatasetOper(ANXTIETY_PHASES_DATASET_PATH)


features_list = []
labels_list = []

time_between_samples_in_seconds = 60

common_dataset_entry_list :List[CommonDatasetEntry]= []
common_dataset_entry_list.extend(anxiety_phases_dataset_oper.retrieve_features_and_labels(reduce_labels=True,selected_task=0,time_between_samples=time_between_samples_in_seconds))
common_dataset_entry_list.extend(anxiety_phases_dataset_oper.retrieve_features_and_labels(reduce_labels=True,selected_task=1,time_between_samples=time_between_samples_in_seconds))

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
    #resultsArr.extend(HrvAnalysisOper.getMaxHour(dataset_entry.rr,dataset_entry.timeline,False))
    resultsArr.extend(HrvAnalysisOper.getHrvFeatures(dataset_entry.rr))

    #resultsArr.extend(HrvAnalysisOper.getHrvAnalysisFeatures(dataset_entry.rr))

    features_list.append(resultsArr)
    labels_list.append(dataset_entry.anxiety_label)

# minmax NO GOOD
#scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = StandardScaler()
features_list = scaler.fit_transform(features_list)

#Save files
np.savetxt("processed_data/panic_dataset_" + time_between_samples_in_seconds.__str__() + "sec_labels.csv", labels_list, delimiter=",")
np.savetxt("processed_data/panic_dataset_" + time_between_samples_in_seconds.__str__() + "sec_features.csv", features_list, delimiter=",")