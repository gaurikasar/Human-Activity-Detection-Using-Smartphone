import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from datetime import datetime
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning


def read_data():
    train = pd.read_csv('UCI_HAR_dataset/csv_files/train.csv')
    test = pd.read_csv('UCI_HAR_dataset/csv_files/test.csv')
    X_train = train.drop(['subject', 'Activity', 'ActivityName'], axis=1)
    y_train = train.ActivityName
    X_test = test.drop(['subject', 'Activity', 'ActivityName'], axis=1)
    y_test = test.ActivityName
    return X_train, y_train, X_test, y_test


def perform_model(model, X_train, y_train, X_test, y_test, class_labels):
    results = dict()
    train_start_time = datetime.now()
    model.fit(X_train, y_train)
    print('Predicting test data')
    test_start_time = datetime.now()
    y_pred = model.predict(X_test)
    results['predicted'] = y_pred
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    results['accuracy'] = accuracy
    print('Accuracy:- {}\n'.format(accuracy))
    cm = metrics.confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    print('\nConfusion Matrix')
    print('\n {}'.format(cm))

    print('----------------Classifiction Report----------------------')
    classification_report = metrics.classification_report(y_test, y_pred)

    results['classification_report'] = classification_report
    print(classification_report)

    results['model'] = model

    return results


def logistic_regression(labels, X_train, y_train, X_test, y_test):
    print("***********Logistic Regression************")
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=ConvergenceWarning)

    parameters = {'C': [0.01, 0.1, 1, 10, 20, 30], 'penalty': ['l2', 'l1']}
    log_reg = linear_model.LogisticRegression()
    log_reg_grid = GridSearchCV(log_reg, param_grid=parameters, cv=3, verbose=1, n_jobs=-1)
    log_reg_grid_results = perform_model(log_reg_grid, X_train, y_train, X_test, y_test, class_labels=labels)


def linear_SVC(labels, X_train, y_train, X_test, y_test):
    print('****************SVC****************')
    parameters = {'C': [0.125, 0.5, 1, 2, 8, 16]}
    lr_svc = LinearSVC(tol=0.00005)
    lr_svc_grid = GridSearchCV(lr_svc, param_grid=parameters, n_jobs=-1, verbose=1)
    lr_svc_grid_results = perform_model(lr_svc_grid, X_train, y_train, X_test, y_test, class_labels=labels)


def SVM(labels, X_train, y_train, X_test, y_test):
    print('*************SVM*********************')
    parameters = {'C': [2, 8, 16], \
                  'gamma': [0.0078125, 0.125, 2]}
    rbf_svm = SVC(kernel='rbf')
    rbf_svm_grid = GridSearchCV(rbf_svm, param_grid=parameters, n_jobs=-1)
    rbf_svm_grid_results = perform_model(rbf_svm_grid, X_train, y_train, X_test, y_test, class_labels=labels)


X_train, y_train, X_test, y_test = read_data()
print('X_train and y_train : ({},{})'.format(X_train.shape, y_train.shape))
print('X_test  and y_test  : ({},{})'.format(X_test.shape, y_test.shape))
labels = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
logistic_regression(labels, X_train, y_train, X_test, y_test)
linear_SVC(labels, X_train, y_train, X_test, y_test)
SVM(labels, X_train, y_train, X_test, y_test)







