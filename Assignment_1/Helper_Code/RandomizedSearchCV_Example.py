# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 11:02:54 2019

@author: Chris
"""

# Example from https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html


print(__doc__) # https://stackoverflow.com/questions/33066383/print-doc-in-python-3-script
 

import numpy as np

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
from sklearn.model_selection import RandomizedSearchCV # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

# get some data
digits = load_digits()
X, y = digits.data, digits.target

# build a classifier
clf = RandomForestClassifier(n_estimators=20)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

#for random decimals the example snippet below is provided.
    #params_randomSearch = {
    #   "learning_rate" : Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
    #   "min_samples_leaf": np.arange(1,30,1)}



# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)

start = time()
random_search.fit(X, y)
print("\n\nRandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings.\n" % ((time() - start), n_iter_search))
report(random_search.cv_results_)




# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
start = time()
grid_search.fit(X, y)

print("\n\nGridSearchCV took %.2f seconds for %d candidate parameter settings.\n"
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)