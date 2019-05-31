# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 13:41:38 2019

@author: Chris
"""

# starter code for GridSearchCV with multiple classifiers. I modified it from the below link.
# https://stackoverflow.com/questions/50265993/alternate-different-models-in-pipeline-for-gridsearchcv


import numpy as np
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# the models that you want to compare
models = {
    'RandomForestClassifier': RandomForestClassifier(),
    'KNeighboursClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression()
}

# the optimisation parameters for each of the above models
params = {
    'RandomForestClassifier':{ 
            "n_estimators"      : [100, 200, 500, 1000],
            "max_features"      : ["auto", "sqrt", "log2"],
            "bootstrap": [True],
            "criterion": ['gini', 'entropy'],
            "oob_score": [True, False]
            },
    'KNeighboursClassifier': {
        'n_neighbors': np.arange(3, 15),
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute']
        },
    'LogisticRegression': {
        'solver': ['newton-cg', 'sag', 'lbfgs'],
        'multi_class': ['ovr', 'multinomial']
        }  
}
    
params2 = {'RandomForestClassifier':{},
           'KNeighboursClassifier':{},
           'LogisticRegression':{}}

#
def fit(train_data, train_target):
        """
        fits the list of models to the training data, thereby obtaining in each 
        case an evaluation score after GridSearchCV cross-validation
        """
        for name in models.keys():
            est = models[name]
            est_params = params2[name]
            gscv = GridSearchCV(estimator=est, param_grid=est_params, cv=5)
            gscv.fit(train_data, train_target)
            print("best parameters are: {}".format(gscv.best_estimator_))
            print("Where we selected the parameters: {}" .format(gscv.cv_results_['params'][gscv.best_index_]))
            print("with mean cross-validated score: {}" .format(gscv.best_score_))
            
# Step 1: bring in your matrix and labels
iris = datasets.load_iris()

# Step 2: Use the fit function above
fit(iris.data, iris.target)

# Step 3: visualize some results with matplotlib...
# Step 4: ?
# Step 5: profit.