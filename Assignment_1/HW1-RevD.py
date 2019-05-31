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

    ## Sample array M includes the X's
    #M = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,14],[13,12],[11,10],[9,8],[7,6],[5,4],[3,2],[1,2],[2,3]])
    #M.shape
    #
    ## Sample array L includes the Y's, values are ones and zeroes (1/3:0 2/3: 1).
    #L = np.random.choice([0,1], size=(M.shape[0],), p=[1./3, 2./3])
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
        print("clf: ", clfString)
        for k1, v1 in clfDict.items():  # go through first level of clfDict
            if k1 in clfString:		# if clfString1 matches first level
                ret_hyper[clf] = [dict(zip(v1, s)) for s in product(*v1.values())]
    return ret_hyper                        

###-------- myClfHypers Function - End  --------####
#################################################### 


# Function Call for parsing hyper parameters from dictionary
hyper_param_dict = myClfHypers(clfsList)

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
        print('\nResults:',results)


        for key in results:
            print('key:',key)
            k1 = results[key]['clf'] 
            v1 = results[key]['accuracy']
            print('accuracy',v1)
            k1Test = str(k1) #Since we have a number of k-folds for each classifier...
                             #We want to prevent unique k1 values due to different "key" values
                             #when we actually have the same classifer and hyper parameter settings.
                             #So, we convert to a string
                            
            #String formatting            
            k1Test = k1Test.replace('            ',' ') # remove large spaces from string
            k1Test = k1Test.replace('          ',' ')
        
        #Then check if the string value 'k1Test' exists as a key in the dictionary
            if k1Test in clfsAccuracyDict:
                print('v1 in append loop',v1)
                clfsAccuracyDict[k1Test].append(v1) #append the values to create an array (techically a list) of values
            else:
                print('v1 in initial loop',v1)
                clfsAccuracyDict[k1Test] = [v1] #create a new key (k1Test) in clfsAccuracyDict with a new value, (v1)
    

for k1, v1 in clfsAccuracyDict.items():  # go through first level of clfDict
    print("\nClassifier with Paramters:",k1,'\nAccuracy',v1,'Mean Accuracy',sum(v1)/len(v1))            


# Determing best median accuracy score for models across (# of kfolds)
best_medianAccuScore = max(statistics.median(v1) for k1, v1 in clfsAccuracyDict.items())
print('\nBest Median Accuracy Score:',best_medianAccuScore)

len(clfsAccuracyDict)
#then for each accuracy in the list... plot the values...

#k1, v1 = zip(*clfsAccuracyDict) # unpack a list of pairs into two tuples


## for determining maximum frequency (# of kfolds) for histogram y-axis
#n = max(len(v1) for k1, v1 in clfsAccuracyDict.items())
#
## for naming the plots
#filename_prefix = 'clf_Histograms_'
#
## initialize the plot_num counter for incrementing in the loop below
#plot_num = 1 
#
## Adjust matplotlib subplots for easy terminal window viewing
#left  = 0.125  # the left side of the subplots of the figure
#right = 0.9    # the right side of the subplots of the figure
#bottom = 0.1   # the bottom of the subplots of the figure
#top = 0.6      # the top of the subplots of the figure
#wspace = 0.2   # the amount of width reserved for space between subplots,
#               # expressed as a fraction of the average axis width
#hspace = 0.2   # the amount of height reserved for space between subplots,
#               # expressed as a fraction of the average axis height

#create the histograms
#matplotlib is used to create the histograms: https://matplotlib.org/index.html
#for k1, v1 in clfsAccuracyDict.items():
#    # for each key in our clfsAccuracyDict, create a new histogram with a given key's values 
#    fig = plt.figure(figsize =(10,10)) # This dictates the size of our histograms
#    ax  = fig.add_subplot(1, 1, 1) # As the ax subplot numbers increase here, the plot gets smaller
#    plt.hist(v1, facecolor='green', alpha=0.75) # create the histogram with the values
#    ax.set_title(k1, fontsize=25) # increase title fontsize for readability
#    ax.set_xlabel('Classifer Accuracy (By K-Fold)', fontsize=25) # increase x-axis label fontsize for readability
#    ax.set_ylabel('Frequency', fontsize=25) # increase y-axis label fontsize for readability
#    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1)) # The accuracy can only be from 0 to 1 (e.g. 0 or 100%)
#    ax.yaxis.set_ticks(np.arange(0, n+1, 1)) # n represents the number of k-folds
#    ax.xaxis.set_tick_params(labelsize=20) # increase x-axis tick fontsize for readability
#    ax.yaxis.set_tick_params(labelsize=20) # increase y-axis tick fontsize for readability
#    #ax.grid(True) # you can turn this on for a grid, but I think it looks messy here.
#
#    # pass in subplot adjustments from above.
#    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
#    plot_num_str = str(plot_num) #convert plot number to string
#    filename = filename_prefix + plot_num_str # concatenate the filename prefix and the plot_num_str
#    plt.savefig(filename, bbox_inches = 'tight') # save the plot to the user's working directory
#    plot_num = plot_num+1 # increment the plot_num counter by 1
#plt.show()



#create the histograms
#matplotlib is used to create the histograms: https://matplotlib.org/index.html
Acc_list = []
Classifier = []
Acc_means = []
Acc_median = []
#type(Acc_list)
for k1, v1 in clfsAccuracyDict.items():
    Acc_list.append(v1)
    Classifier.append(k1)
    Acc_means.append(sum(v1)/len(v1))
    Acc_median.append(statistics.median(v1))

#import seaborn as sns
fig_dims = (len(Acc_list)*1.5, 12)

fig, ax = plt.subplots(figsize=fig_dims)
fig.subplots_adjust(left=0.125, right=0.95, top=0.9, bottom=0.25)
sns.set(style="whitegrid")
ax.set_title("Accuracy Score vs Classification/HyperParameters\nBest Median Accuracy Score: %.4f" % best_medianAccuScore)
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
ax = sns.boxplot(y=Acc_list,x=Classifier)

median_labels = [str(np.round(s, 3)) for s in Acc_median]
pos = range(len(Acc_means))
for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick], Acc_median[tick]+0.001, median_labels[tick],  #medians[tick] + 0.5 instead of 1.001
            horizontalalignment='center', size='medium', color='b', weight='semibold')
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

## for naming the plots
filename = 'clf_Vertical_Boxplots'
plt.savefig(filename, bbox_inches = 'tight') # save the plot to the user's working directory

Acc_list[2]

sum(Acc_list[2])/len(Acc_list[2])







#### Box Horizontal Plot ###

fig_dims = (15, len(clfsAccuracyDict)*1.8)

fig, ax = plt.subplots(figsize=fig_dims)
fig.subplots_adjust(left=0.125, right=0.95, top=0.9, bottom=0.25)
sns.set(style="whitegrid")
ax.set_title("Accuracy Score vs Classification/HyperParameters\nBest Median Accuracy Score: %.4f" % best_medianAccuScore)
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
ax = sns.boxplot(x=Acc_list,y=Classifier)

median_labels = [str(np.round(s, 3)) for s in Acc_median]
pos = range(len(Acc_median))
for tick,label in zip(pos,ax.get_yticklabels()):
    ax.text(Acc_median[tick]+0.001,pos[tick], median_labels[tick],
            verticalalignment='center', size='medium', color='b', weight='semibold',rotation=-90)
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0)

filename = 'clf_Horizontal_Boxplots'
plt.savefig(filename, bbox_inches = 'tight') # save the plot to the user's working directory
