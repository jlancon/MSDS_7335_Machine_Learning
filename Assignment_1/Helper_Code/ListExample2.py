# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 11:11:30 2019

@author: Chris
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold  #EDIT: I had to import KFold 
 
#EDIT: array M includes the X's
M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])

#EDIT: array L includes the Y's, they're all ones and as such is only for example (an ML algorithm would always predict 1).
#L = np.ones(M.shape[0])

#So, I gave us some 0's too for Logistic Regression
L = np.random.choice([0,1], size=(M.shape[0],), p=[1./3, 2./3])
#EDIT: a single value, 5, to use for 5-fold (k-fold) cross validation
n_folds = 5

#EDIT: pack the arrays together into "data"
data = (M,L,n_folds)

#EDIT: Let's see what we have.
print(data)


# data expanded
M, L, n_folds = data
kf = KFold(n_splits=n_folds)

print(kf)


# define the run function
def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data #EDIT: unpack the "data" container of arrays
  kf = KFold(n_splits=n_folds) # JS: Establish the cross validation 
  ret = {} # JS: classic explicaiton of results
  
  for clfs in a_clf: 
      #print(ret) # show this to explain that we have a BUG!
      
      for ids, (train_index, test_index) in enumerate(kf.split(M, L)): #EDIT: We're interating through train and test indexes by using kf.split
                                                                       #      from M and L.
                                                                       #      We're simply splitting rows into train and test rows
                          
          clf = clfs(**clf_hyper) # JS: unpack paramters into clf if they exist   #EDIT: this gives all keyword arguments except 
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

#Use run function

#clfsList = [LogisticRegression, RandomForestClassifier] 
clfsList = [RandomForestClassifier, LogisticRegression] 

results = run(clfsList, data, clf_hyper={})

print(results)





#for clfs in clfsList:
#    results = run(clfs, data, clf_hyper={})
#    print(results)



