# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 17:18:18 2019

@author: Chris
"""

#from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from itertools import product



clfsList = [RandomForestClassifier, LogisticRegression] 


clfDictGoodExample = {'RandomForestClassifier': {"min_samples_split": [2,3,4], 
                                      "n_jobs": [1,2,3]},                                      
                     'LogisticRegression': {"tol": [0.001,0.01,0.1]}}
  

clfDictBadExample = {'RandomForestClassifier': {"min_samples_split": [2,3,4], 
                                      "n_jobs": [1,2,3]},                                      
                     'LogisticRegression': {"tol": [0.001,0.01,0.1]},
                     'SGDClassifier': 'no_k2'} #This will give us problems.
  

# Non-working example:

for k1, v1 in clfDictGoodExample.items(): # go through the inner dictionary of hyper parameters
    #Nothing to do here, we need to get into the inner nested dictionary.
    k2,v2 = zip(*v1.items()) # explain zip
    for values in product(*v2): #for the values in the inner dictionary, get their unique combinations from product()
        hyperSet = dict(zip(k2, values)) # create a dictionary from their values
        print(hyperSet) # print out the results in a dictionary that can be used to feed into the ** operator in run()


# Working Example:
    
for k1, v1 in clfDictBadExample.items(): # go through the inner dictionary of hyper parameters
    #Nothing to do here, we need to get into the inner nested dictionary.
   
    try:
        k2,v2 = zip(*v1.items()) # explain zip
        for values in product(*v2): #for the values in the inner dictionary, get their unique combinations from product()
            hyperSet = dict(zip(k2, values)) # create a dictionary from their values
            print(hyperSet) # print out the results in a dictionary that can be used to feed into the ** operator in run()
    except AttributeError:
        print("no k2 and v2 found")
