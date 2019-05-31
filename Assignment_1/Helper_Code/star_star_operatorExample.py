# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 11:56:07 2019

@author: Chris
"""
import numpy as np
#from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import KFold  #EDIT: I had to import KFold 


#EDIT: array M includes the X's
M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])

L = np.random.choice([0,1], size=(M.shape[0],), p=[1./3, 2./3])

n_folds = 5

#EDIT: pack the arrays together into "data"
data = (M,L,n_folds)


#With no changes to hyper params
simpleHyperDict = {}

clf = RandomForestClassifier(**simpleHyperDict)

print(clf)



# With changes to hyper params
simpleHyperDict = {"min_samples_split": 3, "n_jobs": 2}


clf = RandomForestClassifier(**simpleHyperDict)

print(clf)

