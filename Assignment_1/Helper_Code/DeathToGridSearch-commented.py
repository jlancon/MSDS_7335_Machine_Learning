# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 10:50:56 2019

@author: Chris
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold  #EDIT: I had to import KFold 
 
# adapt this to run 

# Recommend to be done before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parameters for each
 
# Recommend to be done before live class 3
# 2. expand to include larger number of classifiers and hyperparameter settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parameters settings
 
# Recommend to be done before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function


#EDIT: array M includes the X's
M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])

#EDIT: array L includes the Y's, they're all ones and as such is only for example (an ML algorithm would always predict 1).
L = np.ones(M.shape[0])

#EDIT: a single value, 5, to use for 5-fold (k-fold) cross validation
n_folds = 5

#EDIT: pack the arrays together into "data"
data = (M,L,n_folds) #immutable tuple

#EDIT: Let's see what we have.
print(data)


# data expanded
M, L, n_folds = data
kf = KFold(n_splits=n_folds)

print(kf)

#np.set_printoptions(threshold=np.inf)

#EDIT: Show what is kf.split doing
for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    print("k fold = ", ids)
    print("            train indexes", train_index)
    print("            test indexes", test_index)

#EDIT: A function, "run", to run all our classifiers against our data.

def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data #EDIT: unpack the "data" container of arrays
  kf = KFold(n_splits=n_folds) # JS: Establish the cross validation 
  ret = {} # JS: classic explicaiton of results
  
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)): #EDIT: We're interating through train and test indexes by using kf.split
                                                                   #      from M and L.
                                                                   #      We're simply splitting rows into train and test rows
                                                                   #      for our five folds.
    
    clf = a_clf(**clf_hyper) # JS: unpack paramters into clf if they exist   #EDIT: this gives all keyword arguments except 
                                                                             #      for those corresponding to a formal parameter
                                                                             #      in a dictionary.
            
    clf.fit(M[train_index], L[train_index])   #EDIT: First param, M when subset by "train_index", 
                                              #      includes training X's. 
                                              #      Second param, L when subset by "train_index",
                                              #      includes training Y.                             
    
    pred = clf.predict(M[test_index])         #EDIT: Using M -our X's- subset by the test_indexes, 
                                              #      predict the Y's for the test rows.
    
    ret[ids]= {'clf': clf,                    #EDIT: Create arrays of
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}    
  return ret
# Feature importance: For RandomForext predictor
#  print(clf.feature_importances_)

#Use run function
results = run(RandomForestClassifier, data, clf_hyper={})

print(results)


#After explaining.... talk about lists and dictionaries.
#https://docs.python.org/3/tutorial/

#Also... Here's your clfs 
#https://scikit-learn.org/stable/supervised_learning.html

#Go through examples in this order:
# ** operator 
# list1, 
# dictionary, 
# list2