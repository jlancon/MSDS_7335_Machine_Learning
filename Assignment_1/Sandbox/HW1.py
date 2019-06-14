# -*- coding: utf-8 -*-
"""

@author: Jeffrey Lancon
"""

import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import KFold  #EDIT: I had to import KFold 
 
# adapt this to run 

# 1. write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parameters for each
# 2. expand to include larger number of classifiers and hyperparmater settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Collaborate to get things
# 7. Investigate grid search function


#EDIT: array M includes the X's
M = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,14],[13,12],[11,10],[9,8],[7,6],[5,4],[3,2],[1,2],[2,3]])
M.shape

#EDIT: array L includes the Y's, they're all ones and as such is only for example (an ML algorithm would always predict 1).
#L = np.ones(M.shape[0])

#So, I gave us some 0's too for Logistic Regression

#L = np.random.choice([0,1], size=(M.shape[0],), p=[1./3, 2./3])
L = np.random.uniform(low=0.0, high=10, size=(M.shape[0],))
L.shape[0]

#EDIT: a single value, 5, to use for 5-fold (k-fold) cross validation
n_folds = 5

#EDIT: pack the arrays together into "data"
data = (M,L,n_folds)

#EDIT: Let's see what we have.
print(data)


# data expanded
M, L, n_folds = data

kf = KFold(n_splits=n_folds)
kf.get_n_splits()

#??KFold()
print(kf)

#EDIT: Show what is kf.split doing
for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    print("k fold = ", ids)
    print("            train indexes", train_index)
    print("            test indexes", test_index)

#EDIT: A function, "run", to run all our classifiers against our data.

def run(a_regr, data, regr_hyper={}):
  M, L, n_folds = data #EDIT: unpack the "data" container of arrays
  kf = KFold(n_splits=n_folds) # JS: Establish the cross validation 
  ret = {} # JS: classic explicaiton of results
  
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)): #EDIT: We're interating through train and test indexes by using kf.split
                                                                   #      from M and L.
                                                                   #      We're simply splitting rows into train and test rows
                                                                   #      for our five folds.
    
    regr = a_regr(**regr_hyper) # JS: unpack paramters into clf if they exist   #EDIT: this gives all keyword arguments except 
                                                                             #      for those corresponding to a formal parameter
                                                                             #      in a dictionary.
            
    regr.fit(M[train_index], L[train_index])   #EDIT: First param, M when subset by "train_index", 
                                              #      includes training X's. 
                                              #      Second param, L when subset by "train_index",
                                              #      includes training Y.                             
    
    pred = regr.predict(M[test_index])         #EDIT: Using M -our X's- subset by the test_indexes, 
                                              #      predict the Y's for the test rows.
    
    ret[ids]= {'regr': regr,                    #EDIT: Create arrays of
               'train_index': train_index,
               'test_index': test_index,
               'R2_score_regr': regr.score(M[test_index],L[test_index]),
               'R2 Score_': r2_score(L[test_index], pred),
               'L-True': L[test_index],
               'L-Predict' : pred
               
               }    
  return ret

#Use run function with a list and a for loop


#clfsList = [RandomForestRegressor, SVR, SGDRegressor] 
#clfsList = [RandomForestRegressor]
regrsList = [LinearRegression]
for regrs in regrsList:
    results = run(regrs, data, regr_hyper={})
    print(results)
r2_score([4.02315675, 1.32414062, 4.76057163],[3.50946821, 6.6368602 , 6.63405039])
r2_score()