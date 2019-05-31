# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 12:49:19 2019

@author: Chris
"""

#import numpy as np
#from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from itertools import product
#from sklearn.model_selection import KFold  #EDIT: I had to import KFold 

#The official itertools documentation
#https://docs.python.org/3/library/itertools.html


# Very simple example that I modified from:  https://gist.github.com/dhermes/830b2c3fccfda81f3b73

'''
for pair in product((0, 1)):
    print(pair)

for pair in product((0, 1), repeat=2):
    print(pair)
'''


'''
#Itertools lists example

min_samples_split = [2,3,4]
n_jobs = [1,2,3,4]
test = [5,6]

#combos = list(product(min_samples_split, n_jobs))
combos = list(product(min_samples_split, n_jobs, test))

for vals in combos:
    print(vals)  # print out the combinations of values
'''


#Itertools Dictionary Example
clfsList = [RandomForestClassifier, LogisticRegression] 

clfDict = {'RandomForestClassifier': {"min_samples_split": [2,3,4], 
                                      "n_jobs": [1,2,3]},
                                      
           'LogisticRegression': {"tol": [0.001,0.01,0.1]}}


for k1, v1 in clfDict.items(): # go through the inner dictionary of hyper parameters
        #Nothing to do here, we need to get into the inner nested dictionary.
        k2,v2 = zip(*v1.items()) # explain zip (https://docs.python.org/3.3/library/functions.html#zip)
        for values in product(*v2): #for the values in the inner dictionary, get their unique combinations from product()
            hyperSet = dict(zip(k2, values)) # create a dictionary from their values
            print(hyperSet) # print out the results in a dictionary that can be used to feed into the ** operator in run()
