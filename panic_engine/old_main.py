# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from os import cpu_count
from statistics import mean

import numpy as np
from imblearn.over_sampling import SMOTE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from numpy import genfromtxt, std
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

#import heartpy as hp
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from main import evaluate_model

original_labels_list = genfromtxt("./processed_data/big_outliers_short_alllabels_60_suds.csv", delimiter=',', dtype=None, encoding='utf-8')
original_features_list = genfromtxt("./processed_data/big_outliers_short_allnorm_features_60_suds.csv", delimiter=',', dtype=None, encoding='utf-8')


rand =15
oversample = SMOTE(random_state=rand)
print ('random ' + rand.__str__())
features_list, labels_list = oversample.fit_resample(original_features_list, original_labels_list)


sgkf = model_selection.StratifiedKFold(n_splits=3) # 3 splits, true, 42
print("rand_state" + sgkf.__str__())
kfold_train_dataset = []
kfold_train_dataset_labels = []
kfold_test_dataset = []
kfold_test_dataset_labels = []
for _, testIndexes in sgkf.split(original_features_list, original_labels_list):
    trainIndexes = np.delete(range(0, len(features_list)), testIndexes)
    kfold_train_dataset = features_list[trainIndexes]
    kfold_train_dataset_labels = np.take(labels_list,trainIndexes)
    kfold_train_dataset,kfold_train_dataset_labels = oversample.fit_resample(kfold_train_dataset, kfold_train_dataset_labels)

    kfold_test_dataset = original_features_list[testIndexes]
    kfold_test_dataset_labels = np.take(original_labels_list,testIndexes)
    continue



knn1 =  SVC(kernel='rbf',max_iter=100000,C=100,tol=0.001) #(ACC: 0.871)|  0.9512 - split 3
knn2 = knn1
#knn1 = DecisionTreeClassifier(min_samples_leaf=2,min_impurity_decrease=0.02, criterion="entropy",random_state=1,min_samples_split=12)
#knn2 = DecisionTreeClassifier(min_samples_leaf=2,min_impurity_decrease=0.02, criterion="entropy",random_state=1,min_samples_split=12)

#knn1 = AdaBoostClassifier(n_estimators=70) #train ACC: 0.626 - 14 features | Test ACC : 0.75
#knn2 = AdaBoostClassifier(n_estimators=70)
#knn1 = DecisionTreeClassifier(min_samples_leaf=10) 0.68 & 0.73
#knn2 = DecisionTreeClassifier(min_samples_leaf=10)
#knn1 =  KNeighborsClassifier(algorithm='auto',weights='distance',p=1,leaf_size=4, n_neighbors=7) # (ACC: 0.91): 11 feat
#knn1 =  KNeighborsClassifier(algorithm='auto', weights='distance', p=1, metric="cosine", leaf_size=40, n_neighbors=7) # (ACC: 0.91)
#knn2 =  KNeighborsClassifier(algorithm='auto', weights='distance', p=1, metric="cosine", leaf_size=40, n_neighbors=7)
#knn1 =  KNeighborsClassifier() # (ACC: 0.886): (0, 2, 4, 6, 7, 16, 18, 21, 37, 42, 47)
#knn2 =  KNeighborsClassifier()
#knn1 = SVC(kernel='linear') # #train ACC: 0.711 - 14 features | Test ACC : 0.866
#knn1 = LinearSVC(dual=True,intercept_scaling=True,penalty='l2',max_iter=100000) # (ACC: 0.653): 9 features
#knn2 = LinearSVC(dual=True,intercept_scaling=True,penalty='l2',max_iter=100000)
#knn1 = XGBClassifier()
#knn2 = GradientBoostingClassifier()


#knn1 = AdaBoostClassifier(base_estimator=SVC(kernel = 'rbf'),algorithm='SAMME')
#knn2 = AdaBoostClassifier(base_estimator=SVC(kernel = 'rbf'),algorithm='SAMME') $TODO test com o LinearSVC
#knn1 = RandomForestClassifier(n_jobs=-1,min_samples_leaf=10)#(ACC: 0.795): (0, 1, 4, 6, 7, 11, 12, 13, 15) 'knn2__criterion': 'gini', 'knn2__n_estimators': 50,
#knn2 =  RandomForestClassifier(n_jobs=-1,min_samples_leaf=10)
#knn1 = GradientBoostingClassifier() #train ACC: 0.661 - 14 features | Test ACC : 0.583
#knn2  = GradientBoostingClassifier()

min_features_to_select = 5 # Minimum number of features to consider

#RFECV
sfs1 = SFS(knn1,
           forward=True,
           floating=False,
           verbose=2,
           n_jobs=cpu_count(),
           scoring='accuracy')

pipe = Pipeline([('sfs', sfs1),
                 ('knn2', knn2)])

#param_grid = {
#    'sfs__k_features': range (3,16),#range (3,4),
#    'sfs__estimator__n_neighbors':  range(5,12) #range(5,6)
#  }

param_grid = {
    'sfs__k_features': range   (18,33),#range (3,4)
    #'knn2__n_neighbors' :  [2,3,4,5,6,7,8,9],
    #'knn2__n_estimators' :  [10,20,30,50,70,80,100,200],
    #'knn2__C' :  [0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9],
    #'knn2__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    #'knn2__tol' : [1e-2,1e-1,1e-0,1e-3,1e-4]
   # 'sfs__estimator__n_estimators':  range(30,200,10) #range(5,6)
}

gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='accuracy',
                  n_jobs=cpu_count(),
                  refit=False)


#sfs = SequentialFeatureSelector(featureSelector, n_features_to_select=min_features_to_select,cv=model_selection.StratifiedKFold(n_splits=5, shuffle=False),scoring="accuracy",)
#sfs = rfecv = RFECV(estimator=featureSelector,step=1,cv=model_selection.Stra199tifiedKFold(n_splits=8, shuffle=False),
#                    scoring="neg_mean_squared_error", min_features_to_select=min_features_to_select,)
print('heyaaabbbb')
gs.fit(kfold_train_dataset, kfold_train_dataset_labels)
print("Best parameters via GridSearch", gs.best_params_)

pipe.set_params(**gs.best_params_)
pipe.fit(kfold_train_dataset, kfold_train_dataset_labels)

print('\nbest combination (ACC: %.3f): %s\n' % (sfs1.k_score_, sfs1.k_feature_idx_))
print("Best parameters via GridSearch", gs.best_params_)

datasetIndexes= np.asarray(sfs1.k_feature_idx_)
features_list = np.take(features_list,datasetIndexes,axis=1)
#######################
######################
#######################

print('choosen indexes :' + np.array_str(datasetIndexes))

#clf = KNeighborsClassifier(n_neighbors = 8, metric = ' minkowski', p = 2)


#clf = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski', p = 2)

clf =  knn2 # New Best avg 0.7
#clf = KNeighborsClassifier(n_neighbors=8) # BEST SO FAR (0.66)

kfold_predicted_labels = clf.predict(kfold_test_dataset[:,datasetIndexes])
print(classification_report(kfold_test_dataset_labels, kfold_predicted_labels))

#Logger.logPredictionMetrics(kfold_test_dataset_labels,kfold_predicted_labels)

print("#######################################################")
print('features list len' + len(features_list).__str__())
print('test list len' + len(kfold_test_dataset).__str__())


# Press the green button in the gutter to run the script.

X_train = kfold_train_dataset
X_test = kfold_test_dataset
y_train = kfold_train_dataset_labels
y_test = kfold_test_dataset_labels


bestFeatures = list(map(lambda string: int(string.split('x')[1]), fs.fit(X_train, y_train).get_feature_names_out()))
print(classification_report(y_test, kfold_predicted_labels,digits=4))
