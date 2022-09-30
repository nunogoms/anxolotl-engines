
import hrvanalysis
import numpy as np
from matplotlib import pyplot, pyplot as plt
from numpy import genfromtxt
from sklearn import feature_selection
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

possibleFrequencyValues = [1, 10, 20, 30, 45, 60]
featuresDataPath = "processed_data/data/allsets_%dsec_features.csv"
labelsDataPath = "processed_data/data/allsets_%dsec_labels.csv"
fileTime = 30

def main_function():
    printCircularClassDistribution()
    exit(1)
    #printClassDistribution()
    #printAllFeatureImportance()


    titlesArr = ["gender","age","mean_rr","variability_rr","90th_rr","10th_rr","85th-15th_rr"]
    titlesArr.extend(hrvanalysis.get_time_domain_features([230,450]).keys())
    titlesArr.extend(hrvanalysis.get_frequency_domain_features([230, 450]).keys())
    titlesArr.extend(hrvanalysis.get_poincare_plot_features([230, 450]).keys())


    print(titlesArr.__str__())


def printAllFeatureImportance():

    importancesDict ={}

    for time in possibleFrequencyValues:

        original_features_list = genfromtxt(featuresDataPath % time, delimiter=',', dtype=None, encoding='utf-8')
        original_labels_list = genfromtxt(labelsDataPath % time, delimiter=',', dtype=None, encoding='utf-8')
        # get importance
        score = feature_selection.mutual_info_classif(original_features_list,original_labels_list)

        # summarize feature importance
        total_importance = 0
        for v in score:
            total_importance += v
        importancesDict[time]= total_importance


    fig = plt.figure(figsize=(10, len(importancesDict.keys())))

    # creating the bar plot
    plt.bar(importancesDict.keys(), importancesDict.values(),
            width=2)

    plt.xlabel("Time between samples in seconds")
    plt.ylabel("Information Gain")
    plt.title("Information gain per different sampling times")

    plt.show()

def printClassDistribution():
    # create a dataset

    original_labels_list = genfromtxt(labelsDataPath % fileTime, delimiter=',', dtype=None, encoding='utf-8')

    classesSizes = [
        len(np.where(original_labels_list == 0)[0]),
            len(np.where(original_labels_list == 1)[0]),
            len(np.where(original_labels_list == 2)[0])
                ]

    bars = ('0', '1', '2')

    # Create bars with different colors
    fig,ax = plt.subplots()
    plt.bar(bars, classesSizes,
            color=['blue', 'yellow', 'orange'])

    plt.xlabel("Class")
    plt.ylabel("Number of entries")
    plt.title("Label Class Distribution")

    for bars in ax.containers:
        ax.bar_label(bars)

    # Show graph
    plt.show()

def printCircularClassDistribution():
    # create a dataset
    original_labels_list = genfromtxt(labelsDataPath % fileTime, delimiter=',', dtype=None, encoding='utf-8')

    classesSizes = np.asarray([
        len(np.where(original_labels_list == 0)[0]),
            len(np.where(original_labels_list == 1)[0]),
            len(np.where(original_labels_list == 2)[0])
                ])

    label_names = ['No anxiety (Label 0)', 'Average Anxiety (Label 1)', 'High Anxiety (Label 2)']

    # Create bars with different colors
    fig,ax = plt.subplots()

    plt.pie(classesSizes, labels=label_names, colors=['blue', 'yellow', 'orange'])
    #plt.title("Label Class Distribution")

    for label_names in ax.containers:
        ax.bar_label(label_names)

    # Show graph
    plt.show()


def printClassValues():

    importancesDict ={}

    for time in possibleFrequencyValues:

        original_features_list = genfromtxt(featuresDataPath % time, delimiter=',', dtype=None, encoding='utf-8')
        original_labels_list = genfromtxt(labelsDataPath % time, delimiter=',', dtype=None, encoding='utf-8')
        # get importance
        score = feature_selection.mutual_info_classif(original_features_list,original_labels_list)

        # summarize feature importance
        total_importance = 0
        for v in score:
            total_importance += v
        importancesDict[time]= total_importance


    fig = plt.figure(figsize=(10, len(importancesDict.keys())))

    # creating the bar plot
    plt.bar(importancesDict.keys(), importancesDict.values(),
            width=2)

    plt.xlabel("Time between samples in seconds")
    plt.ylabel("Information Gain")
    plt.title("Information gain per different sampling times")

    plt.show()

if __name__ == '__main__':
    main_function()