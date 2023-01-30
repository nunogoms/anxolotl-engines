import math
from typing import List, Dict
import hrvanalysis as hrva
import numpy as np


class HrvAnalysisOper:

    @staticmethod
    def get_max_hour(ibi_list: List, ibi_time_list: List, time_between_samples: int, completeness_percentage:float):
        min_hrv_freq = 0

        best_time_domain_features = None
        best_frequency_domain_features = None
        best_poincare_plot_features = None

        first_time_stamp = ibi_time_list[0]

        step_minutes = 60

        for hour in range(23):
            initial_time = first_time_stamp + (hour * 60 * 60 )
            final_time = first_time_stamp + ((hour + 1) * 60 * 60 )

            hour_indexes = np.argwhere((ibi_time_list >= initial_time) & (ibi_time_list < final_time)).ravel()
            if len(hour_indexes) < ((60/time_between_samples) * step_minutes)*completeness_percentage: continue  # Se tiver menos de 70% do esperado não passa

            ibi_hour_array = np.take(ibi_list, hour_indexes)
            frequency_domain_features = hrva.get_frequency_domain_features(ibi_hour_array)

            if frequency_domain_features['lf_hf_ratio'] > min_hrv_freq: #'hfnu'
                min_hrv_freq = frequency_domain_features['lf_hf_ratio']
            else:
                continue
            best_time_domain_features = hrva.get_time_domain_features(ibi_hour_array)
            best_frequency_domain_features = frequency_domain_features
            best_poincare_plot_features = hrva.get_poincare_plot_features(ibi_hour_array)

        return HrvAnalysisOper._join_hrv_features(best_time_domain_features, best_frequency_domain_features, best_poincare_plot_features)

    @staticmethod
    def _join_hrv_features(time_domain_features: Dict, frequency_domain_features: Dict, poincare_plot_features: Dict):
        results_arr = []

        for entry in time_domain_features.values(): results_arr.append(entry)
        for entry in frequency_domain_features.values(): results_arr.append(entry)
        for entry in poincare_plot_features.values(): results_arr.append(entry)

        return results_arr

    @staticmethod
    def get_hrv_analysis_features(ibi_list: List):
        time_domain_features = hrva.get_time_domain_features(ibi_list)
        frequency_domain_features = hrva.get_frequency_domain_features(ibi_list)
        poincare_plot_features = hrva.get_poincare_plot_features(ibi_list)

        return HrvAnalysisOper._join_hrv_features(time_domain_features, frequency_domain_features, poincare_plot_features)