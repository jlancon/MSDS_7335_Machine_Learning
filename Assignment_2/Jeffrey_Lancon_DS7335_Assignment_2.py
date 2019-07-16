# -*- coding: utf-8 -*-
def header():
    """
    __author__ = "Jeffrey Lancon"
    __course__ = "DS-7335-406 Machine Learning"
    __Assignment__ = "Assignment 2"
    __date__ = "06/30/2019"
    __credits__ = ["Chris Havenstein",""]
    __version__ = "1.0.0"
    __email__ = "jlancon@smu.edu"
    
    Objective: A medical claim is denoted by a claim number ('Claim.Number'). Each
               claim consists of one or more medical lines denoted by a claim line
               number ('Claim.Line.Number').
    
    1. J-codes are procedure codes that start with the letter 'J'.
    
         A. Find the number of claim lines that have J-codes.
         B. How much was paid for J-codes to providers for 'in network' claims?
         C. What are the top five J-codes based on the payment to providers?
    
    2. For the following exercises, determine the number of providers that were 
       paid for at least one J-code. Use the J-code claims for these providers to 
       complete the following exercises.
        A. Create a scatter plot that displays the number of unpaid claims 
           (lines where the ‘Provider.Payment.Amount’ field is equal to zero)
           for each provider versus the number of paid claims.
        B. What insights can you suggest from the graph?
        C. Based on the graph, is the behavior of any of the providers concerning?
    
    3. Consider all claim lines with a J-code.
    
         A. What percentage of J-code claim lines were unpaid?
         B. Create a model to predict when a J-code is unpaid. Explain why you 
            choose the modeling approach.
         C. How accurate is your model at predicting unpaid claims?
          D. What data attributes are predominately influencing the rate of non-payment?
    
    The code should run from terminal and save the results to the directory that
    the program is executed from.
    Output format: Graphics -> .png
                   Data     -> .json   
    
    Platform = ['python3.7']
    Libraries = ['sklearn_0.21.1','numpy_1.16.4','matplotlib_3.1.0',
                     'seaborn_0.9.0']
    
    """
print(header.__doc__)

#import libraries
import numpy as np
import numpy.lib.recfunctions as rfn
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
#import os
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold 
from itertools import product
import statistics
import json
from warnings import simplefilter
from sklearn.preprocessing import OneHotEncoder
#NumPy Cheatsheet - https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)



#https://docs.scipy.org/doc/numpy-1.12.0/reference/arrays.dtypes.html
#These are the data types or dtypes that will be used in the below function, np.genfromtxt()
types = ['S8', 'f8', 'i4', 'i4', 'S14', 'S6', 'S6', 'S6', 'S4', 'S9', 'S7', 'f8',
         'S5', 'S3', 'S3', 'S3', 'S3', 'S3', 'f8', 'f8', 'i4', 'i4', 'i4', 'S3', 
         'S3', 'S3', 'S4', 'S14', 'S14']

#NumPy Structured Arrays: https://docs.scipy.org/doc/numpy/user/basics.rec.html
# Though... I like this Structured Array explanation better in some cases: https://jakevdp.github.io/PythonDataScienceHandbook/02.09-structured-data-numpy.html

#np.genfromtxt:  https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html
#read in the claims data into a structured numpy array
CLAIMS = np.genfromtxt('data\claim.sample.csv', dtype=types, delimiter=',', names=True, 
                       usecols=[0,1,2,3,4,5,6,7,8,9,10,11,
                                12,13,14,15,16,17,18,19,20,21,
                                22,23,24,25,26,27,28])


# Display dtypes and field names
#print(CLAIMS.dtype)
#Notice the shape differs since we're using structured arrays.
#print(CLAIMS.shape)

#String Operations in NumPy - https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.char.html
#Sorting, Searching, and Counting in NumPy - https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.sort.html

np.set_printoptions(threshold=500, suppress=True)


########################################################
# You might see issues here: https://stackoverflow.com/questions/23319266/using-numpy-genfromtxt-gives-typeerror-cant-convert-bytes-object-to-str-impl
# If you do, encode as a unicode byte object
# Idea code for multiple tuples:   http://xion.io/post/code/python-startswith-tuple.html
SCHEME = ('"j'.encode(),'"J'.encode()) #You can use any number of combinations to seach
index = []
for i in range(len(CLAIMS['ProcedureCode'])):
    if CLAIMS['ProcedureCode'][i].startswith(SCHEME):
        index.append(i)

Jcodes = CLAIMS[index]
#print(Jcodes)

###############################################################
#QUESTION: Find the number of claim lines that have J-codes with "Jcodes"?

print('\n\n*** J-Code Procedures ***\n')
print('Number of claim lines that have J-Code ProcedureCodes: ',len(Jcodes))
Total_JCodes = np.sum(Jcodes['ProviderPaymentAmount'])
print('Total Payments for J-Code Procedures (In and Out of Network):  $', round(Total_JCodes,2))


########################################################
#QUESTION: How much was paid for J-codes to providers for 'in network' claims?

SCHEME = ('"I'.encode())
index = []
for i in range(len(Jcodes['InOutOfNetwork'])):
    if Jcodes['InOutOfNetwork'][i].startswith(SCHEME):
        index.append(i)

InNetwork = Jcodes[index]

print('\n\n*** In Network J-Code Payments ***\n')
print('Number of In-Network JCode Procdures: ',len(InNetwork))

InNetworkPayments_JCodes = np.sum(InNetwork['ProviderPaymentAmount'])
print('Total Payments for In-Network J-Code Procedures:  $', round(InNetworkPayments_JCodes,2))

##############################################################

# Sorted Jcodes, by ProviderPaymentAmount; in decending order in 1 step
Sorted_Jcodes = np.sort(Jcodes, order='ProviderPaymentAmount')[::-1]

#https://www.numpy.org/devdocs/user/basics.rec.html
# We want to find the top five J-codes based on the payment to providers?
#Join and flatten subset of Sorted_Jcodes 'ProcedureCode' and 'ProviderPaymentAmount' 
Jcodes_with_ProviderPayments = rfn.merge_arrays((Sorted_Jcodes['ProcedureCode'],Sorted_Jcodes['ProviderPaymentAmount']), flatten = True, usemask = False)


#GroupBy JCodes using a dictionary: Aggregating Provider Payment amounts by Jcode
JCode_dict = {}

#Aggregate with Jcodes - code  modifiedfrom a former student's code, Anthony Schrams
for aJCode in Jcodes_with_ProviderPayments:
    if aJCode[0] in JCode_dict.keys():
        JCode_dict[aJCode[0]] += aJCode[1]
    else:
        aJCode[0] not in JCode_dict.keys()
        JCode_dict[aJCode[0]] = aJCode[1]

# print (JCode_dict)

#sum the JCodes
#np.sum([v1 for k1,v1 in JCode_dict.items()])


#create an OrderedDict (which we imported from collections): https://docs.python.org/3.7/library/collections.html#collections.OrderedDict
#Then, sort in descending order
JCodes_PaymentsAgg_descending = OrderedDict(sorted(JCode_dict.items(), key=lambda aJCode: aJCode[1], reverse=True))
    
#Display the results        
#print(JCodes_PaymentsAgg_descending)

# What are the top five J-codes based on the payment to providers?
print('\n\n***  Top 5 J-codes based on payments to Providers  ***\n')
count = 0
for k,v in JCodes_PaymentsAgg_descending.items():
    if count < 5:
        print('Procedure Code: ',k,' Payment Amount: $', round(v,2))
        count += 1


'''
2. For the following exercises, determine the number of providers that were paid for at least one J-code. Use the J-code claims for these providers to complete the following exercises.
    A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of paid claims.
    B. What insights can you suggest from the graph?
    C. Based on the graph, is the behavior of any of the providers concerning? Explain.
'''


##We need to come up with labels for paid and unpaid Jcodes

###### *******
Unpaid_Jcodes = Sorted_Jcodes[Sorted_Jcodes['ProviderPaymentAmount'] == 0]
Paid_Jcodes = Sorted_Jcodes[Sorted_Jcodes['ProviderPaymentAmount']  > 0]
#len(Unpaid_Jcodes)
#len(Paid_Jcodes)
#len(Unpaid_Jcodes.dtype)
unique, counts = np.unique(Paid_Jcodes['ProviderID'], return_counts=True)
#len(unique)

print('\n\n*** Claims - J-Code Payments ***\n')
print('Total number of "PAID" J-Code Procedure claim lines:        ',len(Paid_Jcodes))
print('Total number of "UN-PAID" J-Code Procedure claim lines:     ',len(Unpaid_Jcodes))
print('Total percentage of "UN-PAID" J-Code Procedure claim lines: ',round((len(Unpaid_Jcodes)/len(Jcodes))*100,2),'%')
print('Total Number of Providers paid for J-Code Procedure:        ',len(unique))

# Need to create labels
# Create a new column and data type for both structured arrays
new_dtype = np.dtype(Unpaid_Jcodes.dtype.descr + [('IsUnpaid', '<i4')])
#len(new_dtype)

#create new structured array with labels
#first get the right shape for each.
Unpaid_Jcodes_w_L = np.zeros(Unpaid_Jcodes.shape, dtype=new_dtype)
Paid_Jcodes_w_L = np.zeros(Paid_Jcodes.shape, dtype=new_dtype)
#Unpaid_Jcodes_w_L.shape
#Paid_Jcodes_w_L.shape
#Unpaid_Jcodes_w_L[0]
#Paid_Jcodes_w_L[0]
#Unpaid_Jcodes_w_L.dtype
#len(Unpaid_Jcodes_w_L.dtype)

#copy the data from Unpaid_Jcodes to Unpaid_JCodes_w_L (with label)
for v1 in Unpaid_Jcodes.dtype.names:
    Unpaid_Jcodes_w_L[v1] = Unpaid_Jcodes[v1]

#And assign the target label to data 
Unpaid_Jcodes_w_L['IsUnpaid'] = 1

# Do the same for the Paid set.
#copy the data from Paid_Jcodes to Paid_JCodes_w_L (with label)
for v1 in Paid_Jcodes.dtype.names:
    Paid_Jcodes_w_L[v1] = Paid_Jcodes[v1]

#And assign the target label 
Paid_Jcodes_w_L['IsUnpaid'] = 0


#now combine the two data sets together by rows (axis=0)
Jcodes_w_L = np.concatenate((Unpaid_Jcodes_w_L, Paid_Jcodes_w_L), axis=0)
#check the shape
#Jcodes_w_L.shape  #Out[46]: (51029,)

# Create merged, flattened, array with 'ProviderID' and 'IsUnpaid' categories
ProviderID_IsUnpaid = rfn.merge_arrays((Jcodes_w_L['ProviderID'],Jcodes_w_L['IsUnpaid']), flatten = True, usemask = False)

#GroupBy JCodes using a dictionary and create an array of 1 and 0 indicating which
# claims were paid '1' an which were not '0'
ProID_IsUnpaid_dict = {}
#Aggregate with Jcodes - code  modified from a former student's code, Anthony Schrams

for aJCode in ProviderID_IsUnpaid:
    if aJCode[0] in ProID_IsUnpaid_dict.keys():
        ProID_IsUnpaid_dict[aJCode[0]].append(int(aJCode[1]))
    else:
        ProID_IsUnpaid_dict[aJCode[0]] = [int(aJCode[1])]


# Unpacking dictionary and summing results to show how many claims were unpaid for each Provider
key_list, paid_list, unpaid_list, unpaid_percentage_list,totalClaims = ([] for i in range(5))
for k, v in ProID_IsUnpaid_dict.items():
    #print(k,len(v))
    key_list.append(k)
    unpaid_list.append(sum(v))
    paid_list.append(len(v)-sum(v))
    unpaid_percentage_list.append(round((((sum(v))/len(v))*100),3))
    totalClaims.append(len(v))

# A. Create a scatter plot that displays the number of unpaid claims
# (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each
# provider versus the number of paid claims.

#Creating color scheme for plot
marker = ['s','o','X','P','D','>','*','s','o','X','P','D','>','*','s']
pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]

fig, ax = plt.subplots(figsize=(9,7))
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 11

ax = sns.scatterplot(x=paid_list, y=unpaid_list,hue=key_list,palette=pkmn_type_colors,s=150)

#for line in range(0,len(paid_list)):
#    ax.plot(paid_list[line],unpaid_list[line],marker=marker[line],markersize=10,color=pkmn_type_colors[line])
#    #add annotations one by one with a loop
#    ax.text(paid_list[line]+25, unpaid_list[line], key_list[line], horizontalalignment='left', size=8, color='black', weight='semibold')
    
ax.legend(bbox_to_anchor=(1,0.65), loc=1)
ax.set_title("Fig.1 - Paid Claims vs Un-Paid Claims by Provider ID")
ax.set(xlabel='Paid Claims', ylabel='Unpaid Claims')

filename = 'Fig_1_Unpaid_vs_Paid'
plt.savefig(filename, bbox_inches = 'tight')
#plt.show()

########################################################################################
###### Creation of Fig 2 - Plot % Un-Paid J-Code Procedures vs Overall J-Code Procedures
fig, ax = plt.subplots(figsize=(9,7))
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 11
ax = sns.scatterplot(x=totalClaims, y=unpaid_percentage_list,hue=key_list,palette=pkmn_type_colors,s=150)

ax.legend(bbox_to_anchor=(1,0.65), loc=1)
ax.set_title("Fig.2 - Percent Unpaid Claims vs Total Claims by Provider ID")
ax.set(xlabel='Total Claims', ylabel='Unpaid Claims as % of Total Claims')

filename = 'Fig_2_PercentUnpaid_vs_Paid'
plt.savefig(filename, bbox_inches = 'tight')
#plt.show()

def Prob2Writeup():
    '''
        B1. Insights can you suggest from the graph (Fig 1)?
           i)   There are 51029 Procedural Codes that begin with 'J'.  Of that, 44961 (88.1%) are un-paid.
                    Investigation as to the reason for this high number of Un-Paid J-Code Procedure claims should be initiated.
           ii) A few providers have an extreme number of unpaid claims(>8000):
                    a) FA0001389001 Total Unpaid Claims: 14842
                    b) FA0001387002 Total Unpaid Claims: 11585
                    c) FA0001387001 Total Unpaid Claims: 8784
            An investigation into why these providers have so many unpaid claims should be initiated (i.e. improper paperwork, procedures, etc.)
        
        C. Based on the graph, is the behavior of any of the providers concerning? Explain.    
        C1. A second graphic was created, looking at the % of unpaid claims vs overall claim
            Based on the percentage graph (Fig 2), is the behavior of any of the providers concerning? Explain.
           i)   The majority of the providers have an average of over 80% un-paid J-Code claims, with the exception of 3 providers:
                        a)  FA1000015001, 61.26%,  Total Claims: 1910 
                        b)  FA1000014001, 51.72%   Total Claims: 1162
                        c)  FA0004551001, 43.69%   Total Claims: 737
                The overall volume of un-paid claims made up by these 3 providers is not exorbidant but we should look into reasons for the high-un-paid claim percentage for these providers.
    '''
print(Prob2Writeup.__doc__)

'''
3. Consider all claim lines with a J-code.

     A. What percentage of J-code claim lines were unpaid?
     B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.
     C. How accurate is your model at predicting unpaid claims?
      D. What data attributes are predominately influencing the rate of non-payment?
'''

########################################################
########################################################

#Prep for machine learning with classifiers in sklearn

########################################################
#We need to shuffle the rows before using classifers in sklearn

# We want to do this since we combined unpaid and paid together, in that order. 
# Displaying observations that overlap the merger location of data set
#Jcodes_w_L[44957:44965]

# Apply the random shuffle, to ensure observations are distributed 
np.random.seed(244)
np.random.shuffle(Jcodes_w_L)

# Displaying observations after shuffle
#Jcodes_w_L[44957:44965]


# recall the features names:
#Jcodes_w_L.dtype.names
#('V1', 'ClaimNumber', 'ClaimLineNumber', 'MemberID', 'ProviderID',
# 'LineOfBusinessID', 'RevenueCode', 'ServiceCode', 'PlaceOfServiceCode',
# 'ProcedureCode', 'DiagnosisCode', 'ClaimChargeAmount', 'DenialReasonCode',
# 'PriceIndex', 'InOutOfNetwork', 'ReferenceIndex', 'PricingIndex', 'CapitationIndex',
# 'SubscriberPaymentAmount', 'ProviderPaymentAmount', 'GroupIndex', 'SubscriberIndex',
# 'SubgroupIndex', 'ClaimType', 'ClaimSubscriberType', 'ClaimPrePrinceIndex',
# 'ClaimCurrentStatus', 'NetworkID', 'AgreementID', 'IsUnpaid')

label =  'IsUnpaid'

# Removed 'V1' and 'Diagnosis Code'
# Removed 'ClaimCurrentStatus', 'DenialReasonCode':   Possible Data Leakage
cat_features = ['ProviderID','LineOfBusinessID','RevenueCode', 
                'ServiceCode', 'PlaceOfServiceCode', 'ProcedureCode',
                'PriceIndex', 'InOutOfNetwork', 'ReferenceIndex', 
                'PricingIndex', 'CapitationIndex', 'ClaimSubscriberType',
                'ClaimPrePrinceIndex', 'NetworkID',
                'AgreementID', 'ClaimType']

# Removed 'ProviderPaymentAmount' since this variable is data leakage
numeric_features = ['ClaimNumber', 'ClaimLineNumber', 'MemberID', 
                    'ClaimChargeAmount',
                    'SubscriberPaymentAmount',
                    'GroupIndex', 'SubscriberIndex', 'SubgroupIndex']

#Displaying on Screen, List of Varibles removed from analysis
print('\n' *4)
print('____________________________________________________________\n\n   ** Variables Removed from Dataset Prior to Analysis **\n____________________________________________________________\n')
print(' Non-Descriptive - Monotonic Variables: \n    "V1" ,\n    "Diagnosis Code"')
print('\n Data Leakage Variables: \n    "ProviderPaymentAmount", \n    "ClaimCurrentStatus,\n    "DenailReasonCode"')
print('\n____________________________________________________________\n')



#convert features to list, then to np.array 
# This step is important for sklearn to use the data from the structured NumPy array

#separate categorical and numeric features
Mcat = np.array(Jcodes_w_L[cat_features].tolist())
Mnum = np.array(Jcodes_w_L[numeric_features].tolist())

L = np.array(Jcodes_w_L[label].tolist())
#Mcat.shape
#Mnum.shape
#L.shape

# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing

# Run the OneHotEncoder: Converting Categorical variables into 1/0 variables
# You can encounter a memory error here in which case, you probably should subset.
ohe = OneHotEncoder(sparse=False) #Easier to read
Mcat = ohe.fit_transform(Mcat)

#If you want to go back to the original mappings.
#ohe.inverse_transform(Mcat)
#ohe_features = ohe.get_feature_names(cat_features).tolist()

#What is the shape of the matrix categorical columns that were OneHotEncoded?   
#Mcat.shape
#Mnum.shape


#You can subset if you have memory issues.

    #If you want to recover from the memory error then subset
    #Mcat = np.array(Jcodes_w_L[cat_features].tolist())
    
    #Mcat_subset = Mcat[0:10000]
    #Mcat_subset.shape
    
    #Mnum_subset = Mnum[0:10000]
    #Mnum_subset.shape
    
    #L_subset = L[0:10000]


#What is the size in megabytes after encoding?
# https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-33.php
# and using base2 (binary conversion), https://www.gbmb.org/bytes-to-mb
#print("%d Megabytes" % ((Mcat.size * Mcat.itemsize)/1048576))
#print("%d Megabytes" % ((Mnum.size * Mnum.itemsize)/1048576))

#Concatenate the columns
M = np.concatenate((Mcat, Mnum), axis=1)

#L = Jcodes_w_L[label].astype(int)

M.shape
L.shape
L[0].dtype

# Now you can use your DeathToGridsearch code.

# Enter: Number of folds (k-fold) cross validation
n_folds = 10

# Creating List of Classifiers to use for analysis
clfsList = [RandomForestClassifier,KNeighborsClassifier] 

# Enter: Range of Hyper-parameters  user wishes to manipulate
# for each classifier listed
# NOTE: No effort was placed on improving Accuracy by manipulating hyper-parameters
# Paramters were chosen as examples only.
clfDict = {'RandomForestClassifier': {
                "min_samples_split": [2,3,5],
                "n_estimators": [10,50],
                "max_depth": [6,10]},'KNeighborsClassifier': {
                "n_neighbors": [2,3,5],
                "algorithm": ['auto','ball_tree']}}

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
               'f1': f1_score(L[test_index], pred)    
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
            v1 = results[key]['f1']
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
    

# Sorting clfsAccuracyDict by median accuracy values
# Returns a sorted list of tuples by median accuracy scores (highest first). Tuples contain Classifer/hyperparameters, k-fold accuracies
# Note: Was not sure if we could utilize 'Collections' package; therefore, chose to use sorted list in lieu of sorted dictionary
clfsAccuracylist_sorted = sorted(clfsAccuracyDict.items(), key=lambda item: statistics.median(item[1]), reverse=True) 


# Displaying Classifiers / Accuracy for each k-fold / and Median Accuracy
for i in clfsAccuracylist_sorted:  # go through first level of clfDict
    print("\nClassifier with Paramters:",i[0],'\nk-Fold F1 Scores',i[1],'\nMedian F1',statistics.median(i[1])) 


#Deconstructing clfsAccuracylist for plotting
F1_list, Classifier, F1_median = ([] for i in range(3))

for i in clfsAccuracylist_sorted:
    F1_list.append(i[1])
    Classifier.append(i[0])
    F1_median.append(statistics.median(i[1]))


#_____________________ Box Plot - Horizontal Orientation -Begin ________________
# BoxPlot for each individual classifier/hyper-parameter combination
# Plot is automatically scaled according to number of classifier/hyper-parameter combination present
# Displays boxplot and median Accuracy for each combination
# Also, exports plot to current directory in png format 
fig_dims = (15, len(clfsAccuracylist_sorted)*1.8)

fig, ax = plt.subplots(figsize=fig_dims)
fig.subplots_adjust(left=0.125, right=0.95, top=0.9, bottom=0.25)
sns.set(style="whitegrid")
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 20
ax.set_title("F-1 Score vs Classification/Hyper-Parameters\nBest Median F-1 Score: {} ".format(round(F1_median[0],4)))
ax.set(xlabel='F1-Score per k-fold', ylabel='Classifier/Parameters')
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
ax = sns.boxplot(x=F1_list,y=Classifier)

median_labels = [str(np.round(s, 4)) for s in F1_median]
pos = range(len(F1_median))
for tick,label in zip(pos,ax.get_yticklabels()):
    ax.text(F1_median[tick],pos[tick], median_labels[tick],
            verticalalignment='center', size='medium', color='black', weight='semibold',rotation=-90)
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0)

filename = 'Classifier_F1_Boxplots'
plt.savefig(filename, bbox_inches = 'tight') # save the plot to the user's working directory
#plt.show()

#_____________________ Box Plot - Horizontal Orientation - End ________________


#Displaying on Screen, the Top 5 median F1 scores for models across classifiers
print('\n' *10)
print('__________________________________________\n\n    ** Top 5 Median F1 Scores **\n__________________________________________\n')
rank = 0
for i in range (5):
    rank = i+1
    print(' Ranking: ',rank,'\n Median F1 Score: ',np.round(F1_median[i],4),'\n Classifier and Parameters:\n   ',Classifier[i],'\n___________________________________________\n')



# Saving classifierAccuracy List to json file, for possible further analysis
json = json.dumps(clfsAccuracylist_sorted)
f = open("ClassifierF1_Score_List.json","w")
f.write(json)
f.close()

############# End of File ########
