# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 18:41:24 2019

@author: Chris
"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold  
from sklearn import datasets
from itertools import product
import matplotlib.pyplot as plt


# Part 1: Bring your data
iris = datasets.load_iris()

# array M includes the X's/Matrix/the data
M = iris.data

# Array L includes Y values/labels/target
L = iris.target

#EDIT: a single value, 5, to use for 5-fold (k-fold) cross validation
n_folds = 5

# pack the arrays together into "data"
data = (M,L,n_folds)


# Part 2: define required functions.. could alternatively create classes... etc..


# A function, "run", to run all our classifiers against our data.

def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data # Unpack the "data" container of arrays
  kf = KFold(n_splits=n_folds) # Establish the cross validation 
  ret = {} #  Classic explication of results
  
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)): #      We're interating through train and test indexes by using kf.split
                                                                   #      from M and L.
                                                                   #      We're simply splitting rows into train and test rows
                                                                   #      for our five folds.
    
    clf = a_clf(**clf_hyper) # unpack parameters into clf if they exist. 
                             # This gives all keyword arguments except for those corresponding to a formal parameter in a dictionary.
                                                                             
            
    clf.fit(M[train_index], L[train_index])   #      First param, M when subset by "train_index", 
                                              #      includes training X's. 
                                              #      Second param, L when subset by "train_index",
                                              #      includes training Y.                             
    
    pred = clf.predict(M[test_index])         #      Using M -our X's- subset by the test_indexes, 
                                              #      predict the Y's for the test rows.
    
    ret[ids]= {'clf': clf,                    #      Create dictionary return object for each k-fold
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}  #Could add other evaluation metrics here...(e.g. precision/recall/f1-score/etc.)  
  return ret



# A dictionary where scores are kept by model and hyper parameter combinations.
def populateClfAccuracyDict(results):
    for key in results:
        k1 = results[key]['clf'] 
        v1 = results[key]['accuracy']
        k1Test = str(k1) #Since we have a number of k-folds for each classifier...
                         #We want to prevent unique k1 values due to different "key" values
                         #when we actually have the same classifer and hyper parameter settings.
                         #So, we convert to a string
                        
        #String formatting            
        k1Test = k1Test.replace('            ',' ') # remove large spaces from string
        k1Test = k1Test.replace('          ',' ')
        
        #Then check if the string value 'k1Test' exists as a key in the dictionary
        if k1Test in clfsAccuracyDict:
            clfsAccuracyDict[k1Test].append(v1) #append the values to create an array (techically a list) of values
        else:
            clfsAccuracyDict[k1Test] = [v1] #create a new key (k1Test) in clfsAccuracyDict with a new value, (v1)            
        
            

def myHyperParamSearch(clfsList,clfDict):
    #hyperSet = {}
    for clf in clfsList:
    
    #I need to check if values in clfsList are in clfDict ... 
    # Note: You could do this without this list.. I did this for teaching purposes.
        clfString = str(clf)
        #print("clf: ", clfString)
        
        for k1, v1 in clfDict.items(): # go through the inner dictionary of hyper parameters
            #Nothing to do here, we need to get into the inner nested dictionary.
            if k1 in clfString:
                #allows you to do all the matching key and values
                k2,v2 = zip(*v1.items()) # explain zip (https://docs.python.org/3.3/library/functions.html#zip)
                for values in product(*v2): #for the values in the inner dictionary, get their unique combinations from product()
                    hyperParams = dict(zip(k2, values)) # create a dictionary from their values
                    results = run(clf, data, hyperParams) # pass the clf and dictionary of hyper param combinations to run; get results
                    populateClfAccuracyDict(results) # populate clfsAccuracyDict with results
 


#Could do this without a clfsList... I just did it for teaching purposes... You'd also restructure the myHyperParamSearch function
clfsList = [RandomForestClassifier, LogisticRegression, KNeighborsClassifier] 

clfDict = {'RandomForestClassifier': {"min_samples_split": [2,3,4], 
                                      "n_jobs": [1,2,3]},                                      
           'LogisticRegression': {"tol": [0.001,0.01,0.1]},           
           'KNeighborsClassifier': {'n_neighbors': np.arange(3, 15),
                                     'weights': ['uniform', 'distance'],
                                     'algorithm': ['ball_tree', 'kd_tree', 'brute']}}

                   
#Declare empty clfs Accuracy Dict to populate in myHyperSetSearch     
clfsAccuracyDict = {}

#Run myHyperSetSearch
myHyperParamSearch(clfsList,clfDict)    

print(clfsAccuracyDict)


# for determining maximum frequency (# of kfolds) for histogram y-axis
n = max(len(v1) for k1, v1 in clfsAccuracyDict.items())

# for naming the plots
filename_prefix = 'clf_Histograms_'

# initialize the plot_num counter for incrementing in the loop below
plot_num = 1 

# Adjust matplotlib subplots for easy terminal window viewing
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.6      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for space between subplots,
               # expressed as a fraction of the average axis width
hspace = 0.2   # the amount of height reserved for space between subplots,
               # expressed as a fraction of the average axis height
               


#create the histograms
for k1, v1 in clfsAccuracyDict.items():
    # for each key in our clfsAccuracyDict, create a new histogram with a given key's values 
    fig = plt.figure(figsize =(20,10)) # This dictates the size of our histograms
    ax  = fig.add_subplot(1, 1, 1) # As the ax subplot numbers increase here, the plot gets smaller
    plt.hist(v1, facecolor='green', alpha=0.75) # create the histogram with the values
    ax.set_title(k1, fontsize=30) # increase title fontsize for readability
    ax.set_xlabel('Classifer Accuracy (By K-Fold)', fontsize=25) # increase x-axis label fontsize for readability
    ax.set_ylabel('Frequency', fontsize=25) # increase y-axis label fontsize for readability
    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1)) # The accuracy can only be from 0 to 1 (e.g. 0 or 100%)
    ax.yaxis.set_ticks(np.arange(0, n+1, 1)) # n represents the number of k-folds
    ax.xaxis.set_tick_params(labelsize=20) # increase x-axis tick fontsize for readability
    ax.yaxis.set_tick_params(labelsize=20) # increase y-axis tick fontsize for readability
    #ax.grid(True) # you can turn this on for a grid, but I think it looks messy here.

    # pass in subplot adjustments from above.
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    plot_num_str = str(plot_num) #convert plot number to string
    filename = filename_prefix + plot_num_str # concatenate the filename prefix and the plot_num_str
    plt.savefig(filename, bbox_inches = 'tight') # save the plot to the user's working directory
    plot_num = plot_num+1 # increment the plot_num counter by 1
    
plt.show()
