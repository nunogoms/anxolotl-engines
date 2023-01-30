import glob
from typing import List

import numpy as np
from numpy import ndarray, genfromtxt
import os
import hrvanalysis as hrva

from dataset_handlers.common_dataset_entry import CommonDatasetEntry


class AnxietyPhasesDatasetEntry:

    def __init__(self, rr: [ndarray], participant_details: ndarray ):
        self.rr = rr
        self.participant_details = participant_details


class AnxietyPhasesDatasetOper:

    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path

    @staticmethod
    def _load_dataset_entry(dataset_entry_path: str,):
        return genfromtxt(dataset_entry_path, delimiter=',', dtype=None, encoding='utf-8')

    @staticmethod
    def load_patient_data(dataset_path: str,task_name:str, patient_name: str) -> AnxietyPhasesDatasetEntry:
        patient_name = patient_name.capitalize()
        patient_file_name: str = "%s_Heart" % patient_name
        hrvfiles_path: str = "%s/%s/%s" % (dataset_path, "electrocardiogram_data",task_name)

        dataset_entry = AnxietyPhasesDatasetEntry(
            rr=AnxietyPhasesDatasetOper._load_dataset_entry("%s/%s" % (hrvfiles_path, patient_file_name + ".csv")),
            participant_details=AnxietyPhasesDatasetOper._load_dataset_entry("%s/%s" % (dataset_path, "participants_details.csv")),
        )

        dataset_entry_row = np.where(dataset_entry.participant_details == patient_name)[0][0]



        dataset_entry.participant_details = np.array([dataset_entry.participant_details[0],dataset_entry.participant_details[dataset_entry_row]])

        return dataset_entry

    @staticmethod
    def subsample_rr_dataset(dataset_rr_entry: ndarray, sample_time: int, overflow_protection: bool = True, normalize_to_seconds: bool = True) -> ndarray:
        indexes_to_keep: List = []
        next_timestamp: float = sample_time
        time_rr_index: int = 0
        over_flow_counter: int = 1  # To take into account delays on the measurements

        for row_index in range(len(dataset_rr_entry)):
            #Convert to seconds from millis
            current_timestamp = int(dataset_rr_entry[row_index][time_rr_index] / 1000)

            if current_timestamp >= next_timestamp:
                indexes_to_keep.append(row_index)
                if over_flow_counter > 0:
                    over_flow_counter -= 1
                    next_timestamp = current_timestamp + (sample_time/2)
                else:
                    if overflow_protection: over_flow_counter = 1
                    next_timestamp = current_timestamp + sample_time

        return dataset_rr_entry[indexes_to_keep]

    @staticmethod
    def map_label(value_to_analyze: float):
        new_label = 1
        if value_to_analyze <= 3:
            new_label = 0
        elif value_to_analyze >= 9:
            new_label = 2
        return new_label

    @staticmethod
    def map_label_suds(value_to_analyze: float):
        new_label = 1
        if value_to_analyze <= 20:
            new_label = 0
        elif value_to_analyze > 60:
            new_label = 2
        return new_label

    @staticmethod
    def create_common_dataset_entry(anxiety_data_entry: AnxietyPhasesDatasetEntry, reduce_labels:bool, label_name: str) -> CommonDatasetEntry:
        gender_index = np.where(anxiety_data_entry.participant_details[0] == 'Gender')[0][0]
        age_index = np.where(anxiety_data_entry.participant_details[0] == 'Age')[0][0]
        anxiety_index = np.where(anxiety_data_entry.participant_details[0] == label_name)[0][0]

        anxiety_label = int(anxiety_data_entry.participant_details[1][anxiety_index])
        if reduce_labels:
            if label_name.lower().__contains__('suds'): anxiety_label = AnxietyPhasesDatasetOper.map_label_suds(anxiety_label)
            else: anxiety_label = AnxietyPhasesDatasetOper.map_label(anxiety_label)

        timeline_arr = anxiety_data_entry.rr[:, 0]
        timeline_arr = np.array(list(map(lambda timestamp : int(timestamp/1000), timeline_arr)))

        patient_common_entry: CommonDatasetEntry = CommonDatasetEntry(
            timeline= timeline_arr,
            rr= anxiety_data_entry.rr[:, 1] ,
            user_age=int(anxiety_data_entry.participant_details[1][age_index].split('-')[0]),
            user_gender=np.where(anxiety_data_entry.participant_details[1][gender_index]=='M', 0,1).max(),
            questionnaire=anxiety_label
        )

        return patient_common_entry

    #Selected task : 0 - speaking task, 1 - bug box task
    #Number users : -1 -> all
    #time_between_samples in seconds
    def retrieve_features_and_labels(self, selected_task:int, reduce_labels: bool, number_users: int = -1, time_between_samples: int = 1) -> ( List[CommonDatasetEntry]):
        common_dataset_entry_list: List = []

        task_to_read: str= ""
        label_name: str = ""
        if selected_task == 0 :
            task_to_read = "speaking_task"
            #label_name = "Speech_Anxiety_Score"
            label_name = "Speech_SUDS"
        elif selected_task == 1 :
            task_to_read = "bug_box_task"
            #label_name = "Bug_Anxiety_Score"
            label_name = "BugBox_Preparation_SUDS"


        hrvfiles_path: str = "%s/%s/%s" % (self.dataset_path, "electrocardiogram_data", task_to_read)
        hrvfile_names = glob.glob("%s/*csv" % hrvfiles_path)

        counter = 0
        for hrvfile in hrvfile_names :


            hrvfile_split = hrvfile.split(os.sep)
            patient_file_name = hrvfile_split[len(hrvfile_split)-1]
            patient_name = patient_file_name.split('_')[0]
            print(patient_file_name )


            patient_data = self.load_patient_data(self.dataset_path, task_to_read, patient_name)
            patient_data.rr = np.array(list(map(lambda val : [val[0],val[1]],patient_data.rr)))

            # This remove outliers from signal
            rr_intervals_without_outliers = hrva.remove_outliers(rr_intervals=patient_data.rr[:, 1], low_rri=180, high_rri=2200)
            # This replace outliers nan values with linear interpolation
            rr_intervals_without_outliers = hrva.interpolate_nan_values(rr_intervals=rr_intervals_without_outliers,interpolation_method="linear")

            #nn_intervals_list = hrva.remove_ectopic_beats(rr_intervals=patient_data.rr[:, 1], method="malik")
            # This replace ectopic beats nan values with linear interpolation


            #rr_intervals_without_outliers = hrva.remove_outliers(rr_intervals=patient_data.rr[:, 1], low_rri=0, high_rri=60000)

            patient_data.rr[:, 1] = np.asarray(rr_intervals_without_outliers)

            patient_data.rr = patient_data.rr[patient_data.rr[:, 1] > 0]
            patient_data.rr = patient_data.rr[np.where(~np.isnan(patient_data.rr[:, 1]))]
            patient_data.rr = self.subsample_rr_dataset(patient_data.rr, sample_time=time_between_samples, overflow_protection=False)

            new_dataset_entry = self.create_common_dataset_entry(patient_data, reduce_labels=reduce_labels, label_name=label_name)
            common_dataset_entry_list.append(new_dataset_entry)

            counter += 1
            if number_users != -1 and number_users <= counter : break

        return  common_dataset_entry_list

        #l1 = self.load_patient_data(ANXTIETY_PHASES_DATASET_PATH, "speaking_task", "P14")
        #nan_filtered_indexes = hrva.remove_outliers(l1.rr[:, 1], low_rri=200, high_rri=2200, verbose=False)

        #l1.rr = l1.rr[np.where(~np.isnan(nan_filtered_indexes))]
        #l1.rr = anxiety_phases_dataset_oper.subsample_rr_dataset(l1.rr, sample_time=1)

