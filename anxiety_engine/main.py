# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from os import cpu_count
from statistics import mean

import hrvanalysis
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from numpy import genfromtxt, std
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

#import heartpy as hp
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def saveModels(pipe, save=False):
    from sklearn2pmml import sklearn2pmml, make_pmml_pipeline
    if not save: return

    models_storage_path = "./models_storage/"

    #needs Java
    # sklearn2pmml(make_pmml_pipeline(pipe), "best_model_95_anxiety.pmml", with_repr = True)

    joblib.dump(pipe, models_storage_path + "newModel.pkl", compress=9)
    #Used on the server to import the model, its working :)
    #model_clone = joblib.load('my_model.pkl')



original_features_list = genfromtxt("processed_data/data/allsets_30sec_features.csv", delimiter=',', dtype=None, encoding='utf-8')
original_labels_list = genfromtxt("processed_data/data/allsets_30sec_labels.csv", delimiter=',', dtype=None, encoding='utf-8')

print("labels 0 :" + len(np.where(original_labels_list == 0)[0]).__str__())
print("labels 1 :" + len(np.where(original_labels_list == 1)[0]).__str__())
print("labels 2 :" + len(np.where(original_labels_list == 2)[0]).__str__())
print("Total :" + len(original_labels_list).__str__())


rand=218
oversample = SMOTE(random_state=rand)
print ('random ' + rand.__str__())
features_list, labels_list = oversample.fit_resample(original_features_list, original_labels_list)

sgkf = model_selection.StratifiedKFold(n_splits=5)
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


# split into train and test sets
X_train = kfold_train_dataset
X_test = kfold_test_dataset
y_train = kfold_train_dataset_labels
y_test = kfold_test_dataset_labels
# feature selection
# define the evaluation method
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# define dataset
# define number of features to evaluate
num_features = [i + 1 for i in range(X_train.shape[1])]
# enumerate each number of features
results = list()
bestFeatures = []
bestScore= 0

"""These two are sacred with
processed_data/data/old_joint_dataset_30sec_features.csv
rand=218
"""
model= SVC(kernel='rbf',max_iter=100000, C = 80, tol=0.5,gamma="auto") #best for newhrv 30 no ectopic - 92% e 82%
#model = RandomForestClassifier(criterion="entropy",random_state=0,n_estimators=200,max_features=None)
#model= LogisticRegression(solver= 'newton-cg', C=400,tol=0.01)
#model= XGBClassifier() #  (ACC: 0.834) |  0.8537
#model = DecisionTreeClassifier( criterion="entropy",random_state=0, min_impurity_decrease=0.001, min_samples_leaf=1, min_samples_split=5)
#model = KNeighborsClassifier(n_neighbors=3) #(ACC: 0.797) - test 0.85

param_grid = {
    'anova__k': range   (20,34),#range (3,4)
}

fs = SelectKBest(score_func=f_classif)
pipe = Pipeline(steps=[('anova', fs), ('model', model)])
gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='accuracy',
                  n_jobs=cpu_count(),
                  refit=False)


print('Searching for best parameters')
gs.fit(kfold_train_dataset, kfold_train_dataset_labels)
print("Best parameters via GridSearch", gs.best_params_)

print('\nTesting with train dataset')
pipe.set_params(**gs.best_params_)
pipe.fit(kfold_train_dataset, kfold_train_dataset_labels)
bestFeatures = list(map(lambda string: int(string.split('x')[1]), fs.fit(X_train, y_train).get_feature_names_out()))
print('best combination (ACC: %.3f): %s\n' % (gs.best_score_, bestFeatures))

trainAcc = gs.best_score_

print('\nTesting with TEST dataset')
pipe.set_params(**gs.best_params_)
pipe.fit( X_train[:,bestFeatures], y_train)
kfold_predicted_labels =pipe.predict(X_test[:,bestFeatures])

print(classification_report(kfold_test_dataset_labels, kfold_predicted_labels,digits=4))

testAcc =accuracy_score(y_true=kfold_test_dataset_labels,y_pred= kfold_predicted_labels)
print("testAcc " + testAcc.__str__()  )

saveModels(pipe)
"""
    if testAcc>= 0.9 and trainAcc >=0.835:
        print(classification_report(kfold_test_dataset_labels, kfold_predicted_labels, digits=4))
        break;
    elif testAcc >= 0.85 and trainAcc >0.82:
        print(classification_report(kfold_test_dataset_labels, kfold_predicted_labels, digits=4))
"""



