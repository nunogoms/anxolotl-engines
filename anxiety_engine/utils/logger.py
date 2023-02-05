import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor


class Logger:
    def __init__(self):
        super().__init__()

    # Let's explore the contents of the dataset directory
    @staticmethod
    def log_prediction_metrics(expected_labels, predicted_labels):
        # Accuracy = TP+TN/TP+FP+FN+TN - accuracy
        # Recall = TP/TP+FN - true positive ratio
        # Precision = TP/TP+FP - all positive ratio
        # F1 Score = 2 * (Recall * Precision) / (Recall + Precision) - better for uneven classes

        print("\n")
        print("######################################")
        print("####### PREDICTION METRICS INFO ######")
        print("######################################\n")

        print(f"acc : {metrics.accuracy_score(expected_labels, predicted_labels).__str__()}")
        # Micro is more accurate, since the class distributions are not equal, as explained in the challenge
        print(f"recall : {metrics.recall_score(expected_labels, predicted_labels, average='weighted').__str__()}")
        print(f"f1 score : {metrics.f1_score(expected_labels, predicted_labels, average='weighted').__str__()}")
        print(f"precision : {metrics.precision_score(expected_labels, predicted_labels, average='weighted',zero_division=0).__str__()}")
        # print(f"roc auc : {metrics.roc_auc_score(expected_labels, predicted_dataset, average='weighted',multi_dclass='ovo').__str__()}")
        print()
        print("############### ERROR ###############")
        print(f"mean squared error  : {metrics.mean_squared_error(expected_labels, predicted_labels).__str__()}")
        print(f"median error  : {metrics.median_absolute_error(expected_labels, predicted_labels).__str__()}")
        print(f"max error  : {metrics.max_error(expected_labels, predicted_labels).__str__()}")
        print()

    @staticmethod
    def log_avg_prediction_metrics(accuracy, f1score):
        # Accuracy = TP+TN/TP+FP+FN+TN - accuracy
        # Recall = TP/TP+FN - true positive ratio
        # Precision = TP/TP+FP - all positive ratio
        # F1 Score = 2 * (Recall * Precision) / (Recall + Precision) - better for uneven classes

        print("\n")
        print("######################################")
        print("##### AVG PREDICTION METRICS INFO ####")
        print("######################################\n")

        print(f"accuracy : {accuracy.__str__()}")
        # Micro is more accurate, since the class distributions are not equal, as explained in the challenge
        print(f"f1 score : {f1score.__str__()}")
        print("\n")

    @staticmethod
    def log_classifier(clf):
        print("\n")
        print("######################################")
        print("########### CLASSIFIER INFO ##########")
        print("######################################\n")
        print(f"type - {clf.__str__()}\n")
        print(f"epsilon - {clf.epsilon.__str__()}")
        print(f"gamma - {clf.gamma.__str__()}")
        print(f"coef0 - {clf.coef0.__str__()}")
        print(f"kernel - {clf.kernel.__str__()}")
        print()

    @staticmethod
    def log_work_duration(start_time, end_time):
        print("\n")
        print("######################################")
        print("######### WORK DURATION INFO #########")
        print("######################################\n")
        print(f"started : {start_time.__str__()}")
        # Micro is more accurate, since the class distributions are not equal, as explained in the challenge
        print(f"ended : {end_time.__str__()}")
        print(f"duration : {end_time.__sub__(start_time).__str__()}")
        print()


class FeatureLogger :
    @staticmethod
    def log_feature_rankings(processed_features, labels):
        # Feature extraction
        model = LogisticRegression()
        rfe = RFE(model)
        fit = rfe.fit(processed_features, labels)
        print("Num Features: %s" % (fit.n_features_))
        print("Selected Features: %s" % (fit.support_))
        print("Feature Ranking: %s" % fit.ranking_)

    @staticmethod
    def log_label_correlation(processed_features, labels):
        # Feature extraction
        ridge = Ridge(alpha=0.00001)
        ridge.fit(processed_features, labels)
        Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
              normalize=False, random_state=None, solver='auto', tol=0.001)

        print("\n\nRidge model:\n", FeatureLogger._pretty_print_coefs(ridge.coef_))

    @staticmethod
    def log_features_correlation(processed_features, algorithm='pearson'):
        reversed_train_dataset = dict()
        for trainArrIndex in range(len(processed_features)):
            for featureTrainArrIndex in range(len(processed_features[trainArrIndex])):
                col_name = 'f' + featureTrainArrIndex.__str__()
                if col_name not in reversed_train_dataset: reversed_train_dataset[col_name] = []
                reversed_train_dataset[col_name].append(processed_features[trainArrIndex][featureTrainArrIndex])

        xyz = pd.DataFrame(reversed_train_dataset)

        corr_matrix = xyz.corr(method=algorithm).round(decimals=2)

        print(corr_matrix)

    def log_features_importance(filtered_dataset, filtered_dataset_labels):
        # define the model
        importance_model = DecisionTreeRegressor()
        # fit the model
        importance_model.fit(filtered_dataset, filtered_dataset_labels)
        # get importance
        importance = importance_model.feature_importances_
        importance = np.sort(importance)
        # summarize feature importance
        for i, v in enumerate(importance):
            # Useful to remove features that wont be used for the model
            print('Feature: %0d, Importance Score: %.5f' % (i, v))

    # A helper method for pretty-printing the coefficients
    @staticmethod
    def _pretty_print_coefs(coefs, names=None, sort=False):
        if names is None:
            names = ["X%s" % x for x in range(len(coefs))]
        lst = zip(coefs, names)
        if sort:
            lst = sorted(lst, key=lambda x: -np.abs(x[0]))
        return "\n".join("Feature %s - %s" % (name, round(coef, 3)) for coef, name in lst)