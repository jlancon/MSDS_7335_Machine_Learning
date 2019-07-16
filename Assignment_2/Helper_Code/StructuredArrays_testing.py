# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 17:31:47 2019

@author: Chris
"""
#import libraries
import numpy as np
import numpy.lib.recfunctions as rfn
from collections import OrderedDict

#NumPy Cheatsheet - https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf


## HW notes:
'''    
A medical claim is denoted by a claim number ('Claim.Number'). Each claim consists of one or more medical lines denoted by a claim line number ('Claim.Line.Number').

1. J-codes are procedure codes that start with the letter 'J'.

     A. Find the number of claim lines that have J-codes.

     B. How much was paid for J-codes to providers for 'in network' claims?

     C. What are the top five J-codes based on the payment to providers?



2. For the following exercises, determine the number of providers that were paid for at least one J-code. Use the J-code claims for these providers to complete the following exercises.

    A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of paid claims.

    B. What insights can you suggest from the graph?

    C. Based on the graph, is the behavior of any of the providers concerning? Explain.



3. Consider all claim lines with a J-code.

     A. What percentage of J-code claim lines were unpaid?

     B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.

     C. How accurate is your model at predicting unpaid claims?

      D. What data attributes are predominately influencing the rate of non-payment?
'''


#Read the two first two lines of the file.
with open('data\claim.sample.csv', 'r') as f:
    print(f.readline())
    print(f.readline())


#Colunn names that will be used in the below function, np.genfromtxt
names = ["V1","Claim.Number","Claim.Line.Number",
         "Member.ID","Provider.ID","Line.Of.Business.ID",
         "Revenue.Code","Service.Code","Place.Of.Service.Code",
         "Procedure.Code","Diagnosis.Code","Claim.Charge.Amount",
         "Denial.Reason.Code","Price.Index","In.Out.Of.Network",
         "Reference.Index","Pricing.Index","Capitation.Index",
         "Subscriber.Payment.Amount","Provider.Payment.Amount",
         "Group.Index","Subscriber.Index","Subgroup.Index",
         "Claim.Type","Claim.Subscriber.Type","Claim.Pre.Prince.Index",
         "Claim.Current.Status","Network.ID","Agreement.ID"]


#https://docs.scipy.org/doc/numpy-1.12.0/reference/arrays.dtypes.html
#These are the data types or dtypes that will be used in the below function, np.genfromtxt()
types = ['S8', 'f8', 'i4', 'i4', 'S14', 'S6', 'S6', 'S6', 'S4', 'S9', 'S7', 'f8',
         'S5', 'S3', 'S3', 'S3', 'S3', 'S3', 'f8', 'f8', 'i4', 'i4', 'i4', 'S3', 
         'S3', 'S3', 'S4', 'S14', 'S14']


#NumPy Structured Arrays: https://docs.scipy.org/doc/numpy/user/basics.rec.html
# Though... I like this Structured Array explanation better in some cases: https://jakevdp.github.io/PythonDataScienceHandbook/02.09-structured-data-numpy.html
#np.genfromtxt:  https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html

import os
os.getcwd()
os.chdir("C:\\Users\\Prodigy\\Documents\\GitRepositories\\MSDS_7335_Machine_Learning\\Assignment_2")
#read in the claims data into a structured numpy array
CLAIMS = np.genfromtxt('data\claim.sample.csv', dtype=types, delimiter=',', names=True, 
                       usecols=[0,1,2,3,4,5,
                                6,7,8,9,10,11,
                                12,13,14,15,16,
                                17,18,19,20,21,
                                22,23,24,25,26,
                                27,28])


#print dtypes and field names
print(CLAIMS.dtype)

#Notice the shape differs since we're using structured arrays.
print(CLAIMS.shape)

#However, you can still subset it to get a specific row.
print(CLAIMS[0])

#Subset it to get a specific value.
print(CLAIMS[0][1])

#Get the names
print(CLAIMS.dtype.names)

#Subset into a column
print(CLAIMS['MemberID'])

#Subset into a column and a row value
print(CLAIMS[0]['MemberID'])


#String Operations in NumPy - https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.char.html

#Sorting, Searching, and Counting in NumPy - https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.sort.html

# You might see issues here: https://stackoverflow.com/questions/23319266/using-numpy-genfromtxt-gives-typeerror-cant-convert-bytes-object-to-str-impl

# If you do, encode as a unicode byte object
#A test string
test = 'J'
test = test.encode()

#A test NumPy array of type string
testStrArray = np.array(['Ja','JA', 'naJ', 'na' ],dtype='S9')

#Showing what the original string array looks like
print('Original String Array: ', testStrArray)

#Now try using startswith()
Test1Indexes = np.core.defchararray.startswith(testStrArray, test, start=0, end=None)
testResult1 = testStrArray[Test1Indexes]

#Showing what the original subset string array looks like with startswith()
print('Subset String Array with startswith(): ', testResult1)

#Now try using find()
TestIndexes = np.flatnonzero(np.core.defchararray.find(testStrArray,test)!=-1)

testResult2 = testStrArray[TestIndexes]

#Showing what the original subset string array looks like with find()
print('Subset String Array with find(): ', testResult2)

#Try startswith() on CLAIMS
JcodeIndexes = np.core.defchararray.startswith(CLAIMS['ProcedureCode'], test, start=1, end=2)

np.set_printoptions(threshold=500, suppress=True)

#Using those indexes, subset CLAIMS to only Jcodes
Jcodes = CLAIMS[JcodeIndexes]

print(Jcodes)


#Try find() on CLAIMS
JcodeIndexes = np.flatnonzero(np.core.defchararray.find(CLAIMS['ProcedureCode'], test, start=1, end=2)!=-1)

#Using those indexes, subset CLAIMS to only Jcodes
Jcodes = CLAIMS[JcodeIndexes]

print(Jcodes)

print(Jcodes.dtype.names)




##############################################################################

#A test NumPy array of type string
SCHEME = ("J".encode(),"j".encode(),"b".encode())
testStrArray = np.array(['Ja','JA', 'naJ', 'ja','cc','j3','3j','bC' ],dtype='S9')
ind = []
for i in range(len(testStrArray)):
    if testStrArray[i].startswith(SCHEME):
        ind.append(i)

testResult3 = testStrArray[ind]
#Showing what the original subset string array looks like with startswith()
print('Original String Array: ', testStrArray)
print('Subset String Array with multiple criteria using startswith(): ', testResult3)

#################################################################

########################################################
SCHEME = ('"j'.encode(),'"J'.encode(),'"b'.encode())
index = []
for i in range(len(CLAIMS['ProcedureCode'])):
    if CLAIMS['ProcedureCode'][i].startswith(SCHEME):
        index.append(i)

Jcodes_test = CLAIMS[index]
print(Jcodes_test)
print(Jcodes_test.dtype.names)
###############################################################

#QUESTION: How do you find the number of claim lines that have J-codes with "Jcodes"?
#You can figure this out. :)
print('\n\n*** J-Code Procedures ***')
print('Number of claim lines that have J-Code ProcedureCodes: ',len(Jcodes_test))
Total_JCodes = np.sum(Jcodes_test['ProviderPaymentAmount'])
print('Total Payments for In-Network J-Code Procedures:  $', round(Total_JCodes,2))

#QUESTION: How much was paid for J-codes to providers for 'in network' claims?
#Give this a try on your own after viewing the example code below.

SCHEME = ('"I'.encode())
index = []
for i in range(len(Jcodes_test['InOutOfNetwork'])):
    if Jcodes_test['InOutOfNetwork'][i].startswith(SCHEME):
        index.append(i)

InNetwork_test = Jcodes_test[index]
#print(InNetwork_test)
#print(InNetwork_test.dtype.names)

print('\n\n*** In Network J-Code Payments ***')
print('Number of In-Network JCode Procdures: ',len(InNetwork_test))

InNetworkPayments_JCodes = np.sum(InNetwork_test['ProviderPaymentAmount'])
print('Total Payments for In-Network J-Code Procedures:  $', round(InNetworkPayments_JCodes,2))
##############################################################

#Sorted Jcodes, by ProviderPaymentAmount
Sorted_Jcodes = np.sort(Jcodes, order='ProviderPaymentAmount')
# Alternate method which sorts in decending order in 1 step
Sorted_Jcodes = np.sort(Jcodes_test, order='ProviderPaymentAmount')[::-1]

# Reverse the sorted Jcodes (A.K.A. in descending order)
Sorted_Jcodes = Sorted_Jcodes[::-1]
# [7, 6, 5, 4, 3, 2, 1]


# What are the top five J-codes based on the payment to providers?
print('\nTop 5 J-codes based on payments to Providers: ')
for i in range(5):
    print('Procedure Code: ',Sorted_Jcodes['ProcedureCode'][i],' Payment Amount: ',Sorted_Jcodes['ProviderPaymentAmount'][i])


# We still need to group the data
print(Sorted_Jcodes[:10])

# You can subset it...
ProviderPayments = Sorted_Jcodes['ProviderPaymentAmount']
Jcodes = Sorted_Jcodes['ProcedureCode']

#recall their data types
Jcodes.dtype
ProviderPayments.dtype

#get the first three values for Jcodes
Jcodes[:3]

#get the first three values for ProviderPayments
ProviderPayments[:3]

#Join arrays together
arrays = [Jcodes, ProviderPayments]

#https://www.numpy.org/devdocs/user/basics.rec.html
Jcodes_with_ProviderPayments = rfn.merge_arrays(arrays, flatten = True, usemask = False)
Jcodes_with_ProviderPayments
Jcodes_with_ProviderPayments2 = rfn.merge_arrays((Sorted_Jcodes['ProcedureCode'],Sorted_Jcodes['ProviderPaymentAmount']), flatten = True, usemask = False)

# What does the result look like?
print(Jcodes_with_ProviderPayments[:3])
print(Jcodes_with_ProviderPayments2[:3]) #Alternate
Jcodes_with_ProviderPayments.shape
Jcodes_with_ProviderPayments2.shape #Alternate

#GroupBy JCodes using a dictionary
JCode_dict = {}

#Aggregate with Jcodes - code  modifiedfrom a former student's code, Anthony Schrams
for aJCode in Jcodes_with_ProviderPayments2:
    if aJCode[0] in JCode_dict.keys():
        if aJCode[0] == '"J9310"'.encode():
            print(aJCode[1])
        JCode_dict[aJCode[0]] += aJCode[1]
    else:
        aJCode[0] not in JCode_dict.keys()
        JCode_dict[aJCode[0]] = aJCode[1]
        if aJCode[0] =='"J9310"'.encode():
            print(aJCode[1])



#sum the JCodes
np.sum([v1 for k1,v1 in JCode_dict.items()])



#create an OrderedDict (which we imported from collections): https://docs.python.org/3.7/library/collections.html#collections.OrderedDict
#Then, sort in descending order
JCodes_PaymentsAgg_descending = OrderedDict(sorted(JCode_dict.items(), key=lambda aJCode: aJCode[1], reverse=True))
    
#print the results        
print(JCodes_PaymentsAgg_descending)

# What are the top five J-codes based on the payment to providers?
print('\nTop 5 J-codes based on payments to Providers: ')
for i in range(5):
    print('Procedure Code: ',JCodes_PaymentsAgg_descending[0],' Payment Amount: ',JCodes_PaymentsAgg_descending[1][i])

print('\nTop 5 J-codes based on payments to Providers: ')
count = 0
for k,v in JCodes_PaymentsAgg_descending.items():
    if count < 5:
        print('Procedure Code: ',k,' Payment Amount: $', round(v,2))
        count += 1






sorted_Jcodes_PPayments_Network = rfn.merge_arrays((Sorted_Jcodes['ProcedureCode'],Sorted_Jcodes['ProviderPaymentAmount'],Sorted_Jcodes['InOutOfNetwork']), flatten = True, usemask = False)
print(sorted_Jcodes_PPayments_Network[:3])

ba = Sorted_Jcodes['InOutOfNetwork']
bb = np.unique(ba,return_counts=True)

sorted_Jcodes_PPayments_Network.names
########################################################
SCHEME = ('"I'.encode())
index = []
for i in range(len(sorted_Jcodes_PPayments_Network)):
    if sorted_Jcodes_PPayments_Network[i][2].startswith(SCHEME):
        index.append(i)

InNetwork = sorted_Jcodes_PPayments_Network[index]
InNetwork = {'ProcedureCode':sorted_Jcodes_PPayments_Network[0], 'ProviderPaymentAmount':sorted_Jcodes_PPayments_Network[1],'InOutOfNetwork':sorted_Jcodes_PPayments_Network[2]}
nn = np.sum(InNetwork['InOutOfNetwork'])

print(InNetwork[:,2].cumsum())
print(Jcodes_test.dtype.names)
print('Number of JCode entries: ',len(Jcodes_test))
###############################################################