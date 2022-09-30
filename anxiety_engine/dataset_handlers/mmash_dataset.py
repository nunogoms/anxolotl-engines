from array import array
from datetime import datetime
from typing import List, Dict

import numpy as np
from numpy import genfromtxt, ndarray

from dataset_handlers.common_dataset_entry import CommonDatasetEntry
import hrvanalysis as hrva


class MmashDatasetEntry:

    def __init__(self, activity: ndarray, questionnaire: ndarray, rr: ndarray, user_info: ndarray):
        # saliva: ndarray, sleep: ndarray, actigraph: ndarray):
        self.activity = activity
        self.questionnaire = questionnaire
        self.rr = rr
        # self.saliva = saliva
        # self.sleep = sleep
        self.user_info = user_info
        # self.actigraph = actigraph


class MmashDatasetOper:

    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path

    @staticmethod
    def _load_dataset_entry(dataset_entry_path: str):
        return genfromtxt(dataset_entry_path, delimiter=',', dtype=None, encoding='utf-8')

    @staticmethod
    def load_patient_data(dataset_path: str, patient_name: str) -> MmashDatasetEntry:
        patient_path: str = "%s/%s/" % (dataset_path, patient_name)

        mmash_dataset_entry = MmashDatasetEntry(
            activity=MmashDatasetOper._load_dataset_entry(patient_path + "Activity.csv"),
            questionnaire=MmashDatasetOper._load_dataset_entry(patient_path + "questionnaire.csv"),
            rr=MmashDatasetOper._load_dataset_entry(patient_path + "RR.csv"),
            # saliva=self._load_dataset(patient_path + "saliva.csv"),
            # sleep=self._load_dataset(patient_path + "sleep.csv"),
            user_info=MmashDatasetOper._load_dataset_entry(patient_path + "user_info.csv"),
            # actigraph=self._load_dataset(patient_path + "Actigraph.csv"),
        )

        return mmash_dataset_entry

    @staticmethod
    def normalize_mmash_array(dataset_array: ndarray) -> (array, ndarray):
        titles_arr: array[str] = [''] * len(dataset_array[0])
        result_arr: ndarray[float] = np.zeros((dataset_array.size, len(dataset_array[0])), dtype=float)

        # normalizing titles
        for rowIndex in range(len(dataset_array[0])):
            if rowIndex == 0:
                titles_arr[rowIndex] += 'index'
            else:
                titles_arr[rowIndex] += dataset_array[0][rowIndex].lower()

        epoch_timestamp: datetime = datetime(1970, 1, 1)
        # normalizing values
        for entryIndex in range(1, len(dataset_array)):
            for rowIndex in range(len(dataset_array[0])):

                # Case of being a time type
                if isinstance(dataset_array[entryIndex][rowIndex], str):
                    decoded_str: str = dataset_array[entryIndex][rowIndex]
                    if decoded_str.__contains__(':'):

                        # Meaning it is just hour and minutes (HH:MM)
                        if len(decoded_str) < 6:
                            if decoded_str == '24:00': decoded_str = '00:00'
                            new_date = datetime.strptime(decoded_str, "%H:%M").replace(
                                year=epoch_timestamp.year)
                            result_arr[entryIndex][rowIndex] = (new_date - epoch_timestamp).total_seconds()
                        # Hour minutes and seconds (HH:MM:SS)
                        else:
                            new_date = datetime.strptime(decoded_str, "%H:%M:%S").replace(
                                year=epoch_timestamp.year)
                            result_arr[entryIndex][rowIndex] = (new_date - epoch_timestamp).total_seconds()
                    # Case of gender
                    elif decoded_str.__contains__('M') or decoded_str.__contains__('F'):
                        if decoded_str.__contains__('M'):
                            result_arr[entryIndex][rowIndex] = float(0)
                        else:
                            result_arr[entryIndex][rowIndex] = float(1)
                    # Being a number, which was decoded as string
                    elif len(decoded_str) > 0:
                        result_arr[entryIndex][rowIndex] = float(decoded_str)
                    else:
                        result_arr[entryIndex][rowIndex] = np.NAN
                else:
                    # if dataset_array[entryIndex][rowIndex] == -29 : result_arr[entryIndex][rowIndex] = float(2)
                    result_arr[entryIndex][rowIndex] = float(dataset_array[entryIndex][rowIndex])

        # Remove empty rows
        result_arr = result_arr[~np.all(result_arr == 0, axis=1)]
        return titles_arr, result_arr

    @staticmethod
    def removeDatasetArrayNan(dataset_array: ndarray) -> ndarray:

        indexes_to_remove = []

        for i in range(len(dataset_array)):
            # these are the labels, which should not be removed
            if i == 0:
                continue
            elif np.isnan(dataset_array[i]).any():
                indexes_to_remove.append(i)

        indexes_to_keep = range(len(dataset_array))
        if len(indexes_to_remove) > 0: indexes_to_keep = np.delete(range(len(dataset_array)),
                                                                   np.unique(indexes_to_remove))

        return dataset_array[indexes_to_keep]

    @staticmethod
    def normalize_mmash_dataset(dataset_array: [ndarray], remove_nans: bool = True) -> List[List]:
        return_list: List[List] = []
        for dataset_entry in dataset_array:
            title_arr, values_arr = MmashDatasetOper.normalize_mmash_array(dataset_entry)
            if remove_nans: values_arr = MmashDatasetOper.removeDatasetArrayNan(values_arr)
            return_list.append([title_arr, values_arr])

        return return_list

    @staticmethod
    def remove_activity_entries(dataset_activity_entry: List[List], dataset_rr_entry: List[List],
                                activities_to_remove: List) -> List[List]:
        indexes_to_remove = []
        # Remove 4-light movement, 5- movement, 6 heavy movement
        ## ACTIVITY
        activity_index: int = dataset_activity_entry[0].index('activity')
        start_activity_index: int = dataset_activity_entry[0].index('start')
        end_activity_index: int = dataset_activity_entry[0].index('end')
        day_activity_index: int = dataset_activity_entry[0].index('day')

        ## RR
        time_rr_index: int = dataset_rr_entry[0].index('time')
        day_rr_index: int = dataset_rr_entry[0].index('day')

        for activity_row in dataset_activity_entry[1]:
            if activities_to_remove.__contains__(activity_row[activity_index]):
                starting_activity_time: float = activity_row[start_activity_index]
                ending_activity_time: float = activity_row[end_activity_index]
                day_activity_time: float = activity_row[day_activity_index]

                for rr_row_index in range(len(dataset_rr_entry[1])):
                    rr_row = dataset_rr_entry[1][rr_row_index]
                    if rr_row[day_rr_index] == day_activity_time:
                        if starting_activity_time <= rr_row[time_rr_index] <= ending_activity_time:
                            indexes_to_remove.append(rr_row_index)
                    else:
                        continue

        indexes_to_keep = range(len(dataset_rr_entry[1]))
        if len(indexes_to_remove) > 0: indexes_to_keep = np.delete(range(len(dataset_rr_entry[1])),
                                                                   np.unique(indexes_to_remove))

        return [dataset_rr_entry[0], dataset_rr_entry[1][indexes_to_keep]]

    @staticmethod
    def remove_outliers(dataset_entry: List[List], column_name: str, low_rri: float, high_rri: float) -> List[List]:

        indexes_to_keep = []

        ######################
        column_index: int = dataset_entry[0].index(column_name)

        for row_index in range(len(dataset_entry[1])):
            dataset_entry[1][row_index][column_index] *= 1000
            row = dataset_entry[1][row_index]
            if row[column_index] > low_rri and row[column_index] < high_rri:
                indexes_to_keep.append(row_index)

        return [dataset_entry[0], dataset_entry[1][indexes_to_keep]]

    # Sample time in seconds
    @staticmethod
    def subsample_rr_dataset(dataset_rr_entry: List[List], sample_time: int, overflow_protection: bool = False) -> List[
        List]:
        indexes_to_keep: List = []
        next_timestamp: float = sample_time
        time_rr_index: int = dataset_rr_entry[0].index('time')
        day_rr_index: int = dataset_rr_entry[0].index('day')
        current_day: int = -1
        over_flow_counter: int = 1  # To take into account delays on the measurements

        for row_index in range(len(dataset_rr_entry[1])):
            row = dataset_rr_entry[1][row_index]
            if current_day < row[day_rr_index]:
                current_day = row[day_rr_index]
                indexes_to_keep.append(row_index)
                next_timestamp = row[time_rr_index] + sample_time
                continue

            if row[time_rr_index] >= next_timestamp:
                indexes_to_keep.append(row_index)
                if over_flow_counter > 0:
                    over_flow_counter -= 1
                else:
                    if overflow_protection: over_flow_counter = 1
                    next_timestamp = row[time_rr_index] + sample_time

        return [dataset_rr_entry[0], dataset_rr_entry[1][indexes_to_keep]]

    @staticmethod
    def datasetArrayToDict(dataset_titles: array, dataset_array: ndarray) -> Dict[str, ndarray]:
        dataset_float_dict: Dict[str, ndarray] = {}

        for entryIndex in range(len(dataset_titles)):
            dict_entry: str = dataset_titles[entryIndex]
            dict_value_list: list = [i[entryIndex] for i in dataset_array]

            # Affecting the dict
            dataset_float_dict[dict_entry.lower()] = np.array(dict_value_list)

        return dataset_float_dict

    @staticmethod
    def generate_features_dictionaries(dataset_arrays: [ndarray]) -> (Dict[str, ndarray], List):
        datasetFeaturesDict = []
        datasetResultsDict: Dict[str, List] = {'ibi': [], 'gender': [], 'ibi_time': []}
        datasetLabels: List = []
        for dataset_entry in dataset_arrays:
            datasetFeaturesDict.append(MmashDatasetOper.datasetArrayToDict(dataset_entry[0], dataset_entry[1]))

        final_index = 0;
        for i in range(np.min(datasetFeaturesDict[1]['day']).__int__(),
                       np.max(datasetFeaturesDict[1]['day'] + 1).__int__()):
            initial_index = final_index
            split_index = np.count_nonzero(datasetFeaturesDict[1]['day'] == i)
            final_index = initial_index + split_index
            datasetResultsDict['ibi'].append(datasetFeaturesDict[1]['ibi_s'][initial_index: final_index])
            datasetResultsDict['ibi_time'].append(
                datasetFeaturesDict[1]['time'][initial_index: initial_index + split_index])
            datasetLabels.append([datasetFeaturesDict[3][('stai%i' % i)][0]])

        datasetResultsDict['gender'].append(datasetFeaturesDict[2]['gender'][0])

        return datasetResultsDict, datasetLabels

    # Mapping anxiety labels from MMASH
    @staticmethod
    def map_label(value_to_analyze: float):
        new_label = 1
        if value_to_analyze <= 30:
            new_label = 0
        elif value_to_analyze >= 50:
            new_label = 2
        return new_label

    # time_between_samples in seconds
    def retrieve_features_and_labels(self, reduce_labels: bool, first_user: int = 1, last_user: int = 23,
                                     time_between_samples=60) -> (List[CommonDatasetEntry]):
        common_dataset_entry_list: List = []

        for i in range(first_user, last_user):
            print("user " + i.__str__())
            dataset_patient1_file = self.load_patient_data(self.dataset_path, 'user_%i' % i)
            dataset_features_list: List[List] = self.normalize_mmash_dataset(
                [dataset_patient1_file.activity, dataset_patient1_file.rr, dataset_patient1_file.user_info,
                 dataset_patient1_file.questionnaire])
            dataset_features_list[1] = self.remove_activity_entries(
                dataset_activity_entry=dataset_features_list[0],
                dataset_rr_entry=dataset_features_list[1],
                activities_to_remove=[4, 5, 6])
            # activity_dict = datasetArrayToDict(normalizedActivityTitles,normalizedActivityArr)
            # This remove outliers from signal
            dataset_features_list[1][1][:, 1] *= 1000

            rr_intervals_without_outliers = hrva.remove_outliers(rr_intervals=dataset_features_list[1][1][:, 1],
                                                                 low_rri=180, high_rri=2200)

            # This replace outliers nan values with linear interpolation
            interpolated_rr_intervals = hrva.interpolate_nan_values(rr_intervals=rr_intervals_without_outliers,
                                                                    interpolation_method="linear")


            dataset_features_list[1][1][:, 1] = np.asarray(interpolated_rr_intervals)



            #nn_intervals_list = hrva.remove_ectopic_beats(rr_intervals=dataset_features_list[1][1][:, 1],method="malik")
            # This replace ectopic beats nan values with linear interpolation
            # dataset_features_list[1][1] = dataset_features_list[1][1][np.where(~np.isnan( nn_intervals_list))]

            dataset_features_list[1][1] = dataset_features_list[1][1][dataset_features_list[1][1][:, 1] > 0]
            dataset_features_list[1][1] = dataset_features_list[1][1][np.where(~np.isnan(dataset_features_list[1][1][:, 1]))]

            dataset_features_list[1] = self.subsample_rr_dataset(dataset_rr_entry=dataset_features_list[1],
                                                                 sample_time=time_between_samples,
                                                                 overflow_protection=False)
            features_dict, labels_dict = self.generate_features_dictionaries(dataset_features_list)

            for results_index in range(len(features_dict['ibi'])):
                anxiety_label = labels_dict[results_index][0]
                if reduce_labels: anxiety_label = self.map_label(anxiety_label)

                common_dataset_entry_list.append(CommonDatasetEntry(timeline=features_dict['ibi_time'][results_index],
                                                                    rr=features_dict['ibi'][results_index],
                                                                    user_gender=features_dict['gender'][0],
                                                                    user_age=dataset_features_list[2][1][0][4],
                                                                    questionnaire=anxiety_label))

        return common_dataset_entry_list
