# This is a sample Python script.
import os
from datetime import datetime
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows,
# actions, and settings.
from os import cpu_count

import joblib
import numpy as np
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from numpy import genfromtxt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, \
    cross_val_score
from sklearn.pipeline import Pipeline

from utils import paths

# import heartpy as hp

######      LOADING ENVIRONMENT CONFIGS      ######
load_dotenv(dotenv_path=f"{paths.get_project_root()}/{paths.MAIN_CONFIG_PATH}")

# Train models information

save_train_model_flag_env = os.environ.get("SAVE_TRAINED_MODELS_FLAG").lower() == "true"
train_model_filename_env = os.environ.get("TRAINED_MODEL_FILENAME")
train_model_storage_path = os.environ.get("TRAINED_MODELS_STORAGE_PATH")

# Features array information

input_features_array_path = os.environ.get("INPUT_FEATURES_ARRAY_PATH")
input_features_values_filename = os.environ.get("INPUT_FEATURES_VALUES_FILENAME")
input_features_labels_filename = os.environ.get("INPUT_FEATURES_LABELS_FILENAME")


def save_models(pipe, save=False):
    if not save: return

    # needs Java
    # sklearn2pmml(make_pmml_pipeline(pipe), "best_model_95_anxiety.pmml",
    # with_repr = True)

    if not os.path.isdir(train_model_storage_path):
        os.mkdir(train_model_storage_path)

    joblib.dump(pipe, f"{train_model_storage_path}/"
                      f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}_"
                      f"{train_model_filename_env}",
                compress=9)  # # Used on the server to import the model,
    # its working :)  # model_clone = joblib.load('my_model.pkl')


original_features_list_path = f"{input_features_array_path}/{input_features_values_filename}"

original_labels_list_path = f"{input_features_array_path}/{input_features_labels_filename}"

original_features_list = genfromtxt(original_features_list_path, delimiter=',', dtype=None,
    encoding='utf-8')
original_labels_list = genfromtxt(original_labels_list_path, delimiter=',', dtype=None,
    encoding='utf-8')

print("labels 0 :" + len(np.where(original_labels_list == 0)[0]).__str__())
print("labels 1 :" + len(np.where(original_labels_list == 1)[0]).__str__())

print("Total :" + len(original_labels_list).__str__())

# Versao antiga labels 2 count -9

# 19 não é mau

rand = 20
oversample = SMOTE(random_state=rand)
print('random ' + rand.__str__())
features_list, labels_list = oversample.fit_resample(original_features_list,
                                                     original_labels_list)

sgkf = model_selection.StratifiedKFold(n_splits=4)
print("rand_state" + sgkf.__str__())
kfold_train_dataset = []
kfold_train_dataset_labels = []
kfold_test_dataset = []
kfold_test_dataset_labels = []
for trainIndexes, testIndexes in sgkf.split(features_list, labels_list):
    kfold_train_dataset = features_list[trainIndexes]
    kfold_train_dataset_labels = np.take(labels_list, trainIndexes)

    kfold_test_dataset = features_list[testIndexes]
    kfold_test_dataset_labels = np.take(labels_list, testIndexes)
    continue

# split into train and test sets
x_train = kfold_train_dataset
x_test = kfold_test_dataset
y_train = kfold_train_dataset_labels
y_test = kfold_test_dataset_labels


# feature selection
# define the evaluation method
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1,
                             error_score='raise')
    return scores


# define dataset
# define number of features to evaluate
num_features = [i + 1 for i in range(x_train.shape[1])]
# enumerate each number of features
results = list()
best_features = []
bestScore = 0

"""These two are sacred with
processed_data/data/jointdataset_30sec_features.csv
rand=218
"""

# model= SVC(max_iter=10000000, C=75, tol=0.05,kernel='rbf',gamma='auto')
# #best for newhrv 30 no ectopic - 92% e 82%
model = RandomForestClassifier(n_estimators=800, criterion='entropy',
                               max_features='sqrt', random_state=0,
                               min_samples_split=4)

param_grid = {'anova__k': range(20, 34),  # range (3,4)

              }

fs = SelectKBest(score_func=f_classif)
pipe = Pipeline(steps=[('anova', fs), ('model', model)])
gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy',
                  n_jobs=cpu_count(), refit=False)

print('Searching for best parameters')
gs.fit(kfold_train_dataset, kfold_train_dataset_labels)
print("Best parameters via GridSearch", gs.best_params_)

print('\nTesting with train dataset')
pipe.set_params(**gs.best_params_)
pipe.fit(kfold_train_dataset, kfold_train_dataset_labels)
best_features = list(map(lambda string: int(string.split('x')[1]),
                         fs.fit(x_train, y_train).get_feature_names_out()))
print('best combination (ACC: %.3f): %s\n' % (gs.best_score_, best_features))

trainAcc = gs.best_score_

print('\nTesting with TEST dataset')
pipe.set_params(**gs.best_params_)
pipe.fit(x_train[:, best_features], y_train)
kfold_predicted_labels = pipe.predict(x_test[:, best_features])

print(classification_report(kfold_test_dataset_labels, kfold_predicted_labels,
                            digits=4))

test_acc = accuracy_score(y_true=kfold_test_dataset_labels,
                          y_pred=kfold_predicted_labels)
print("test_acc " + test_acc.__str__())

save_models(pipe)
"""
    if testAcc>= 0.9 and trainAcc >=0.835:
        print(classification_report(kfold_test_dataset_labels, kfold_predicted_labels, digits=4))
        break;
    elif testAcc >= 0.85 and trainAcc >0.82:
        print(classification_report(kfold_test_dataset_labels, kfold_predicted_labels, digits=4))
"""
