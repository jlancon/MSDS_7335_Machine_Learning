# -*- coding: utf-8 -*-
"""

@author: Jeffrey Lancon

1. write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parameters for each
2. expand to include larger number of classifiers and hyperparmater settings
3. find some simple data
4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
5. Please set up your code to be run and save the results to the directory that its executed from
6. Collaborate to get things
7. Investigate grid search function

python 3.6

"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold 
from itertools import product
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
import statistics


# Part 1: Loading Data set - Breast Cancer Data Set within sklearn
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer
cancer = datasets.load_breast_cancer()

# array M includes the X's/Matrix/the data
M = cancer.data
#M.shape

# Array L includes Y values/labels/target
L = cancer.target
#L.shape[0]


# Enter: Number of folds (k-fold) cross validation
n_folds = 10

# Creating List of Classifiers to use for analysis
clfsList = [RandomForestClassifier, LogisticRegression, KNeighborsClassifier] 

# Enter: Range of Hyper-parameters  user wishes to manipulate
# for each classifier listed
clfDict = {'RandomForestClassifier': {
                "min_samples_split": [2,3,4],
                "n_jobs": [1,2]},
            'LogisticRegression': {"tol": [0.001,0.01,0.1]},
            'KNeighborsClassifier': {
                "n_neighbors": [2,3,5,10,25],
                "algorithm": ['auto','ball_tree','brute'],
                "p": [1,2]}}


#Pack the arrays together into "data"
data = (M,L,n_folds)

#Printing Out Data Values
#print(data)

###------------ run Function - Begin ------------####
#####################################################
# Takes variables a_clf (Classifier function); data (X-data set, Y-classification,
# and, number of folds for CV); and clf_hyper (hyper-parameters for classifier)
# Creates folds and .fits model.
# Returns parameters, train/test data, and accuracy score.

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
               'accuracy': accuracy_score(L[test_index], pred)    
               }    
  return ret

###------------ run Function - End   ------------####
#####################################################




###-------- myClfHypers Function -Begin --------####
####################################################
# Takes Classifier hyper-parameter dictionary and creates all-possible
# combinations of hyper-parameter 

def myClfHypers(clfsList):
    ret_hyper = dict();
    for clf in clfsList:
        clfString = str(clf) #Check if values in clfsList are in clfDict
        #print("clf: ", clfString)
        for k1, v1 in clfDict.items():  # go through first level of clfDict
            if k1 in clfString:		# if clfString1 matches first level
                ret_hyper[clf] = [dict(zip(v1, s)) for s in product(*v1.values())]
    return ret_hyper                        

###-------- myClfHypers Function - End  --------####
#################################################### 


# Function Call for parsing hyper parameters from dictionary
hyper_param_dict = myClfHypers(clfsList)
#print(hyper_param_dict)

# Function Call for fitting model with given Classifier and hyper-parameters, 
# using provided Data and n_fold CV
clfsAccuracyDict = {}
results={}
for clfs in clfsList:
    for i in hyper_param_dict[clfs]:
        #print('Classifier:',clfs,'Parameters:',i)
        clf_hyper=i
        #print('\n\nClassifier:',clfs)#,'\nParameters:',clf_hyper)
        results = run(clfs, data, clf_hyper)
        #print('\nResults:',results)


        for key in results:
            #print('key:',key)
            k1 = results[key]['clf'] 
            v1 = results[key]['accuracy']
            #print('accuracy',v1)
            k1Test = str(k1) #Since we have a number of k-folds for each classifier...
                             #We want to prevent unique k1 values due to different "key" values
                             #when we actually have the same classifer and hyper parameter settings.
                             #So, we convert to a string
                            
            #String formatting            
            k1Test = k1Test.replace('            ',' ') # remove large spaces from string
            k1Test = k1Test.replace('          ',' ')
        
        #Then check if the string value 'k1Test' exists as a key in the dictionary
            if k1Test in clfsAccuracyDict:
                #print('v1 in append loop',v1)
                clfsAccuracyDict[k1Test].append(v1) #append the values to create an array (techically a list) of values
            else:
                #print('v1 in initial loop',v1)
                clfsAccuracyDict[k1Test] = [v1] #create a new key (k1Test) in clfsAccuracyDict with a new value, (v1)
    
# Determing best median accuracy score for models across (# of kfolds)
best_medianAccuScore = max(statistics.median(v1) for k1, v1 in clfsAccuracyDict.items())
best_clf_score=()
best_medianAccuScore = 0.0
for k1,v1 in clfsAccuracyDict.items():
    if best_medianAccuScore < statistics.median(v1):
        best_medianAccuScore = statistics.median(v1)
        best_clf_score = k1,best_medianAccuScore

# Displaying Classifiers/Accuracy for each k-fold/ and Median Accuracy
for k1, v1 in clfsAccuracyDict.items():  # go through first level of clfDict
    print("\nClassifier with Paramters:",k1,'\nAccuracy',v1,'\nMedian Accuracy',statistics.median(v1)) 


#Deconstructing ClfAccuracyDict for plotting
Acc_list = []
Classifier = []
Acc_median = []
#type(Acc_list)
for k1, v1 in clfsAccuracyDict.items():
    Acc_list.append(v1)
    Classifier.append(k1)
    Acc_median.append(statistics.median(v1))

#### Box Horizontal Plot ###

fig_dims = (15, len(clfsAccuracyDict)*1.8)

fig, ax = plt.subplots(figsize=fig_dims)
fig.subplots_adjust(left=0.125, right=0.95, top=0.9, bottom=0.25)
sns.set(style="whitegrid")
plt.rcParams["axes.titlesize"] = 35
plt.rcParams["axes.labelsize"] = 30
ax.set_title("Accuracy Score vs Classification/HyperParameters\nBest Median Accuracy Score: {} ".format(round(best_clf_score[1],4)))
ax.set(xlabel='Percent Accuracy per k-fold', ylabel='Classifier/Parameters')
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
ax = sns.boxplot(x=Acc_list,y=Classifier)

median_labels = [str(np.round(s, 4)) for s in Acc_median]
pos = range(len(Acc_median))
for tick,label in zip(pos,ax.get_yticklabels()):
    ax.text(Acc_median[tick]+0.001,pos[tick], median_labels[tick],
            verticalalignment='center', size='medium', color='b', weight='semibold',rotation=-90)
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0)

filename = 'clf_Horizontal_Boxplots'
plt.savefig(filename, bbox_inches = 'tight') # save the plot to the user's working directory

############################################################################

# Displaying best median accuracy score for models across classifiers
print('\n*******\nBest Median Accuracy Score:',best_clf_score[1],'\nClassifier and Parameters:\n',best_clf_score[0],'\n********')
