import math
from typing import List, Dict
import hrvanalysis as hrva
import numpy as np


class HrvAnalysisOper:

    @staticmethod
    def getHrvFeatures(ibi_list: List):

        ibi_hour_array = ibi_list

        best_time_domain_features = hrva.get_time_domain_features(ibi_hour_array)
        best_frequency_domain_features = hrva.get_frequency_domain_features(ibi_hour_array)
        best_poincare_plot_features = hrva.get_poincare_plot_features(ibi_hour_array)

        return HrvAnalysisOper._join_hrv_features(best_time_domain_features, best_frequency_domain_features, best_poincare_plot_features)

    @staticmethod
    def _join_hrv_features(time_domain_features: Dict, frequency_domain_features: Dict, poincare_plot_features: Dict):
        resultsArr = []

        for entry in time_domain_features.values(): resultsArr.append(entry)
        for entry in frequency_domain_features.values(): resultsArr.append(entry)
        for entry in poincare_plot_features.values(): resultsArr.append(entry)

        return resultsArr

    @staticmethod
    def getHrvAnalysisFeatures(ibi_list: List):
        time_domain_features = hrva.get_time_domain_features(ibi_list)
        frequency_domain_features = hrva.get_frequency_domain_features(ibi_list)
        poincare_plot_features = hrva.get_poincare_plot_features(ibi_list)

        return HrvAnalysisOper._join_hrv_features(time_domain_features, frequency_domain_features, poincare_plot_features)