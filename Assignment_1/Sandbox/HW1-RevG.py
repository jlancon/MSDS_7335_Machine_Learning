# -*- coding: utf-8 -*-
'''
__author__ = "Jeffrey Lancon"
__course__ = "DS-7335-406 Machine Learning"
__Assignment__ = "Assignment 1"
__date__ = "06/01/2019"
__credits__ = ["Chris Havenstein",""]
__version__ = "1.0.0"
__email__ = "jlancon@smu.edu"

Objective: Write python code / functions that take a list or dictionary of 
Classifiers and hyper-parameters, i.e. logistic regression, etc.., each with 
three (3) different sets of hyper-parameters, with varying values of each
hyper-parameter and perform analysis for all combinations (?)Classifier X 
(3) hyper-parameters X (?)hyper-parameter values.
Find a simple classification data set and generate.  Fit models to data set,
using k-fold cross validation, and record classifier/hyper-parameter combination,
and model performance criteria results.
Generate graphical represenations (matplotlib/seaborn) that will assist in
identifying the optimal clf/hyper parameter settings.
The code should run from terminal and save the results to the directory that
the program is executed from.
Output format: Graphics -> .png
               Data     -> .json   

Investigate grid search function: 

Platform = ['python3.7']
Libraries = ['sklearn_0.21.1','numpy_1.16.4','matplotlib_3.1.0',
                 'seaborn_0.9.0']

'''
print(__doc__)
#import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold 
from sklearn import datasets
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import json
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)



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
# NOTE: No effort was placed on improving Accuracy by manipulating hyper-parameters
# Paramters were chosen as examples only.
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

# Function Call for fitting model with given Classifier and hyper-parameter combination, 
# using provided Data and n_fold CV, producing a dictionary containing Classifier, and 
# corresponding fold accuracies
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
    

# Sorting clfsAccuracyDict by median accuracy values.  
# Returns a sorted list of tuples by median accuracy scores (highest first). Tuples contain Classifer/hyperparameters, k-fold accuracies
clfsAccuracylist_sorted = sorted(clfsAccuracyDict.items(), key=lambda item: statistics.median(item[1]), reverse=True) 


# Displaying Classifiers/Accuracy for each k-fold/ and Median Accuracy
for i in clfsAccuracylist_sorted:  # go through first level of clfDict
    print("\nClassifier with Paramters:",i[0],'\nk-Fold Accuracy',i[1],'\nMedian Accuracy',statistics.median(i[1])) 


#Deconstructing clfsAccuracylist for plotting
Acc_list, Classifier, Acc_median = ([] for i in range(3))

for i in clfsAccuracylist_sorted:
    Acc_list.append(i[1])
    Classifier.append(i[0])
    Acc_median.append(statistics.median(i[1]))


#_____________________ Box Plot - Horizontal Orientation -Begin ________________
# BoxPlot for each individual classifier/hyper-parameter combination
# Plot is automatically scaled according to number of classifier/hyper-parameter combination present
# Displays boxplot and median Accuracy for each combination
# Also, exports plot to current directory in png format 
fig_dims = (15, len(clfsAccuracylist_sorted)*1.8)

fig, ax = plt.subplots(figsize=fig_dims)
fig.subplots_adjust(left=0.125, right=0.95, top=0.9, bottom=0.25)
sns.set(style="whitegrid")
plt.rcParams["axes.titlesize"] = 35
plt.rcParams["axes.labelsize"] = 30
ax.set_title("Accuracy Score vs Classification/Hyper-Parameters\nBest Median Accuracy Score: {} ".format(round(Acc_median[0],4)))
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

#_____________________ Box Plot - Horizontal Orientation - End ________________


#Displaying on Screen, the Top 5 median accuracy score for models across classifiers
print('\n' *10)
print('__________________________________________\n\n    ** Top 5 Median Accuracy Scores **\n__________________________________________\n')
rank = 0
for i in range (5):
    rank = i+1
    print(' Ranking: ',rank,'\n Median Accuracy Score: ',np.round(Acc_median[i],4),'\n Classifier and Parameters:\n   ',Classifier[i],'\n___________________________________________\n')



# Saving classifierAccuracy List to json file, for possible further analysis
json = json.dumps(clfsAccuracylist_sorted)
f = open("clfAccuracylist.json","w")
f.write(json)
f.close()

############# End of File ########
