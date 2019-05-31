# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 12:49:19 2019

@author: Chris
"""

#import numpy as np
#from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import KFold  #EDIT: I had to import KFold 

clfsList = [RandomForestClassifier, LogisticRegression] 

clfDict = {'RandomForestClassifier': {"min_samples_split": [2,3,4], "n_jobs": [1,2,3]}, 'LogisticRegression': {"tol": [0.001,0.01,0.1]}}


def myClfHypers(clfsList):
    
    for clf in clfsList:
    
    #I need to check if values in clfsList are in clfDict
        clfString = str(clf)
        print("clf: ", clfString)
        
        for k1, v1 in clfDict.items():  # go through first level of clfDict
            if k1 in clfString:		# if clfString1 matches first level
                for k2,v2 in v1.items(): # go through the inner dictionary of hyper parameters
                    print(k2)			 # for each hyper parameter in the inner list..	
                    for vals in v2:		 # go through the values for each hyper parameter 
                        print(vals)		 # and show them...
                        
                        #pdb.set_trace()

myClfHypers(clfsList)