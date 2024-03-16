# Models

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV





def help():
    print('Models with grid search and stratified K-folds...')



# Support Vector Machine
def grid_SVC(X_train, y_train, performance_metric='f1'):
    model = SVC()
    C = np.linspace(0.01 , 1, 3)
    kernels = ['linear', 'poly',]
    gamma = ['scale']
    grid = dict(C = C, kernel = kernels, gamma = gamma)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv,
                           scoring=performance_metric,error_score='raise')
    grid_result = grid_search.fit(X_train, y_train)
    return  grid_result.best_estimator_



# For large data sets SVM
def grid_linearSVC(X_train, y_train, performance_metric='f1'):
    model = LinearSVC()
    penalty = ["l1", "l2"]
    C = np.linspace(0.01 , 1, 3)
    grid = dict(C = C, penalty = penalty)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv,
                           scoring=performance_metric,error_score='raise')
    grid_result = grid_search.fit(X_train, y_train)
    return  grid_result.best_estimator_



# K-Nearest Neighborhood
def grid_KNN(X_train, y_train, performance_metric='f1'):
    model = KNeighborsClassifier()
    n_neighbors = np.arange(1,20,1) # from K to N in steps of z
    weights = ["uniform", "distance"]
    grid = dict(n_neighbors = n_neighbors, weights = weights)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv,
                           scoring=performance_metric,error_score='raise')
    grid_result = grid_search.fit(X_train, y_train)
    return  grid_result.best_estimator_


# Logistic Regression
def grid_lr(X_train, y_train):
    model = LogisticRegression(random_state=666, max_iter=1000)
    class_weight =  [{0:0.05, 1:0.95}, {0:0.1, 1:0.9}, {0:0.2, 1:0.8}]
    solvers = ['liblinear']
    penalty = ['l2','l1']
    c_values = [ 10, 1.0, 0.1, 0.01, 0.001, ]
    grid = dict(solver=solvers,penalty=penalty,C=c_values, class_weight= class_weight)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv,
                           scoring='f1',error_score='raise')
    grid_result = grid_search.fit(X_train, y_train)
    return  grid_result.best_estimator_
