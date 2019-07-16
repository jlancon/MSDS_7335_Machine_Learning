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
#with open('data\claim.sample.csv', 'r') as f:
#    print(f.readline())
#    print(f.readline())


#Colunn names that will be used in the below function, np.genfromtxt
#names = ["V1","Claim.Number","Claim.Line.Number",
#         "Member.ID","Provider.ID","Line.Of.Business.ID",
#         "Revenue.Code","Service.Code","Place.Of.Service.Code",
#         "Procedure.Code","Diagnosis.Code","Claim.Charge.Amount",
#         "Denial.Reason.Code","Price.Index","In.Out.Of.Network",
#         "Reference.Index","Pricing.Index","Capitation.Index",
#         "Subscriber.Payment.Amount","Provider.Payment.Amount",
#         "Group.Index","Subscriber.Index","Subgroup.Index",
#         "Claim.Type","Claim.Subscriber.Type","Claim.Pre.Prince.Index",
#         "Claim.Current.Status","Network.ID","Agreement.ID"]


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
#print(CLAIMS.dtype)

#Notice the shape differs since we're using structured arrays.
#print(CLAIMS.shape)

#However, you can still subset it to get a specific row.
#print(CLAIMS[0])

#Subset it to get a specific value.
#print(CLAIMS[0][1])

#Get the names
#print(CLAIMS.dtype.names)

#Subset into a column
#print(CLAIMS['MemberID'])

#Subset into a column and a row value
#print(CLAIMS[0]['MemberID'])


#String Operations in NumPy - https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.char.html

#Sorting, Searching, and Counting in NumPy - https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.sort.html

# You might see issues here: https://stackoverflow.com/questions/23319266/using-numpy-genfromtxt-gives-typeerror-cant-convert-bytes-object-to-str-impl

# If you do, encode as a unicode byte object
#A test string
#test = 'J'
#test = test.encode()

#A test NumPy array of type string
#testStrArray = np.array(['Ja','JA', 'naJ', 'na' ],dtype='S9')

#Showing what the original string array looks like
#print('Original String Array: ', testStrArray)

#Now try using startswith()
#Test1Indexes = np.core.defchararray.startswith(testStrArray, test, start=0, end=None)
#testResult1 = testStrArray[Test1Indexes]

#Showing what the original subset string array looks like with startswith()
#print('Subset String Array with startswith(): ', testResult1)

#Now try using find()
#TestIndexes = np.flatnonzero(np.core.defchararray.find(testStrArray,test)!=-1)

#testResult2 = testStrArray[TestIndexes]

#Showing what the original subset string array looks like with find()
#print('Subset String Array with find(): ', testResult2)

#Try startswith() on CLAIMS
#JcodeIndexes = np.core.defchararray.startswith(CLAIMS['ProcedureCode'], test, start=1, end=2)

np.set_printoptions(threshold=500, suppress=True)

#Using those indexes, subset CLAIMS to only Jcodes
#Jcodes = CLAIMS[JcodeIndexes]

#print(Jcodes)


#Try find() on CLAIMS
#JcodeIndexes = np.flatnonzero(np.core.defchararray.find(CLAIMS['ProcedureCode'], test, start=1, end=2)!=-1)

#Using those indexes, subset CLAIMS to only Jcodes
#Jcodes = CLAIMS[JcodeIndexes]

#print(Jcodes)

#print(Jcodes.dtype.names)




##############################################################################

##A test NumPy array of type string
#SCHEME = ("J".encode(),"j".encode(),"b".encode())
#testStrArray = np.array(['Ja','JA', 'naJ', 'ja','cc','j3','3j','bC' ],dtype='S9')
#ind = []
#for i in range(len(testStrArray)):
#    if testStrArray[i].startswith(SCHEME):
#        ind.append(i)
#
#testResult3 = testStrArray[ind]
##Showing what the original subset string array looks like with startswith()
#print('Original String Array: ', testStrArray)
#print('Subset String Array with multiple criteria using startswith(): ', testResult3)
#
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
print('Total Payments for J-Code Procedures (In and Out of Network):  $', round(Total_JCodes,2))

#QUESTION: How much was paid for J-codes to providers for 'in network' claims?
#Give this a try on your own after viewing the example code below.

########################################################
SCHEME = ('"I'.encode())
index = []
for i in range(len(Jcodes_test['InOutOfNetwork'])):
    if Jcodes_test['InOutOfNetwork'][i].startswith(SCHEME):
        index.append(i)

InNetwork_test = Jcodes_test[index]
#print(InNetwork_test)
#print(InNetwork_test.dtype.names)

print('\n\n*** In Network J-Code Payments ***\n')
print('Number of In-Network JCode Procdures: ',len(InNetwork_test))

InNetworkPayments_JCodes = np.sum(InNetwork_test['ProviderPaymentAmount'])
print('Total Payments for In-Network J-Code Procedures:  $', round(InNetworkPayments_JCodes,2))
##############################################################

#Sorted Jcodes, by ProviderPaymentAmount
#Sorted_Jcodes = np.sort(Jcodes, order='ProviderPaymentAmount')
# Alternate method which sorts in decending order in 1 step
Sorted_Jcodes = np.sort(Jcodes_test, order='ProviderPaymentAmount')[::-1]

# Reverse the sorted Jcodes (A.K.A. in descending order)
#Sorted_Jcodes = Sorted_Jcodes[::-1]
# [7, 6, 5, 4, 3, 2, 1]


# What are the top five J-codes based on the payment to providers?
#print('\nTop 5 J-codes based on payments to Providers: ')
#for i in range(5):
#    print('Procedure Code: ',Sorted_Jcodes['ProcedureCode'][i],' Payment Amount: ',Sorted_Jcodes['ProviderPaymentAmount'][i])


# We still need to group the data
#print(Sorted_Jcodes[:10])

# You can subset it...
#ProviderPayments = Sorted_Jcodes['ProviderPaymentAmount']
#Jcodes = Sorted_Jcodes['ProcedureCode']

#recall their data types
#Jcodes.dtype
#ProviderPayments.dtype

#get the first three values for Jcodes
#Jcodes[:3]

#get the first three values for ProviderPayments
#ProviderPayments[:3]

#Join arrays together
#arrays = [Jcodes, ProviderPayments]

#https://www.numpy.org/devdocs/user/basics.rec.html
#Jcodes_with_ProviderPayments = rfn.merge_arrays(arrays, flatten = True, usemask = False)
#Jcodes_with_ProviderPayments
Jcodes_with_ProviderPayments = rfn.merge_arrays((Sorted_Jcodes['ProcedureCode'],Sorted_Jcodes['ProviderPaymentAmount']), flatten = True, usemask = False)

# What does the result look like?
#print(Jcodes_with_ProviderPayments[:3])
print(Jcodes_with_ProviderPayments[:3]) #Alternate
#Jcodes_with_ProviderPayments.shape
Jcodes_with_ProviderPayments.shape #Alternate

#GroupBy JCodes using a dictionary
JCode_dict = {}

#Aggregate with Jcodes - code  modifiedfrom a former student's code, Anthony Schrams
for aJCode in Jcodes_with_ProviderPayments:
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
#print(JCodes_PaymentsAgg_descending)

# What are the top five J-codes based on the payment to providers?
#print('\nTop 5 J-codes based on payments to Providers: ')
#for i in range(5):
#    print('Procedure Code: ',JCodes_PaymentsAgg_descending[0],' Payment Amount: ',JCodes_PaymentsAgg_descending[1][i])

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



########################################################

#Prep for machine learning with classifiers in sklearn

########################################################


#print(Sorted_Jcodes.dtype.names)

##We need to come up with labels for paid and unpaid Jcodes

## find unpaid row indexes  
unpaid_mask = (Sorted_Jcodes['ProviderPaymentAmount'] == 0)

## find paid row indexes
paid_mask = (Sorted_Jcodes['ProviderPaymentAmount'] > 0)


#Here are our
Unpaid_Jcodes = Sorted_Jcodes[unpaid_mask]
len(Unpaid_Jcodes)

Paid_Jcodes = Sorted_Jcodes[paid_mask]
len(Paid_Jcodes)

###### *******
Unpaid_Jcodes_test = Sorted_Jcodes[Sorted_Jcodes['ProviderPaymentAmount'] == 0]
Paid_Jcodes_test = Sorted_Jcodes[Sorted_Jcodes['ProviderPaymentAmount']  > 0]
len(Unpaid_Jcodes_test)
len(Paid_Jcodes_test)


#These are still structured numpy arrays
print(Unpaid_Jcodes.dtype.names)
print(Unpaid_Jcodes[0])
print(Unpaid_Jcodes_test.dtype.names)
print(Unpaid_Jcodes_test[0])


print(Paid_Jcodes.dtype.names)
print(Paid_Jcodes[0])

#Now I need to create labels
print(Paid_Jcodes.dtype.descr)
print(Unpaid_Jcodes.dtype.descr)

#create a new column and data type for both structured arrays
new_dtype1 = np.dtype(Unpaid_Jcodes.dtype.descr + [('IsUnpaid', '<i4')])
#new_dtype2 = np.dtype(Paid_Jcodes.dtype.descr + [('IsUnpaid', '<i4')])

print(new_dtype1)
#print(new_dtype2)

#create new structured array with labels

#first get the right shape for each.
Unpaid_Jcodes_w_L = np.zeros(Unpaid_Jcodes.shape, dtype=new_dtype1)
Paid_Jcodes_w_L = np.zeros(Paid_Jcodes.shape, dtype=new_dtype1)
Unpaid_Jcodes_w_L_test = np.zeros(Unpaid_Jcodes_test.shape, dtype=new_dtype1)
Paid_Jcodes_w_L_test = np.zeros(Paid_Jcodes_test.shape, dtype=new_dtype1)


#check the shape
Unpaid_Jcodes_w_L.shape
Paid_Jcodes_w_L.shape
Unpaid_Jcodes_w_L_test.shape
Paid_Jcodes_w_L_test.shape

#Look at the data
print(Unpaid_Jcodes_w_L)
print(Paid_Jcodes_w_L)
print(Unpaid_Jcodes_w_L_test)
print(Paid_Jcodes_w_L_test)


for v1 in Unpaid_Jcodes_test.dtype.names:
    Unpaid_Jcodes_w_L_test[v1] = Unpaid_Jcodes_test[v1]



#copy the data
Unpaid_Jcodes_w_L['V1'] = Unpaid_Jcodes['V1']
Unpaid_Jcodes_w_L['ClaimNumber'] = Unpaid_Jcodes['ClaimNumber']
Unpaid_Jcodes_w_L['ClaimLineNumber'] = Unpaid_Jcodes['ClaimLineNumber']
Unpaid_Jcodes_w_L['MemberID'] = Unpaid_Jcodes['MemberID']
Unpaid_Jcodes_w_L['ProviderID'] = Unpaid_Jcodes['ProviderID']
Unpaid_Jcodes_w_L['LineOfBusinessID'] = Unpaid_Jcodes['LineOfBusinessID']
Unpaid_Jcodes_w_L['RevenueCode'] = Unpaid_Jcodes['RevenueCode']
Unpaid_Jcodes_w_L['ServiceCode'] = Unpaid_Jcodes['ServiceCode']
Unpaid_Jcodes_w_L['PlaceOfServiceCode'] = Unpaid_Jcodes['PlaceOfServiceCode']
Unpaid_Jcodes_w_L['ProcedureCode'] = Unpaid_Jcodes['ProcedureCode']
Unpaid_Jcodes_w_L['DiagnosisCode'] = Unpaid_Jcodes['DiagnosisCode']
Unpaid_Jcodes_w_L['ClaimChargeAmount'] = Unpaid_Jcodes['ClaimChargeAmount']
Unpaid_Jcodes_w_L['DenialReasonCode'] = Unpaid_Jcodes['DenialReasonCode']
Unpaid_Jcodes_w_L['PriceIndex'] = Unpaid_Jcodes['PriceIndex']
Unpaid_Jcodes_w_L['InOutOfNetwork'] = Unpaid_Jcodes['InOutOfNetwork']
Unpaid_Jcodes_w_L['ReferenceIndex'] = Unpaid_Jcodes['ReferenceIndex']
Unpaid_Jcodes_w_L['PricingIndex'] = Unpaid_Jcodes['PricingIndex']
Unpaid_Jcodes_w_L['CapitationIndex'] = Unpaid_Jcodes['CapitationIndex']
Unpaid_Jcodes_w_L['SubscriberPaymentAmount'] = Unpaid_Jcodes['SubscriberPaymentAmount']
Unpaid_Jcodes_w_L['ProviderPaymentAmount'] = Unpaid_Jcodes['ProviderPaymentAmount']
Unpaid_Jcodes_w_L['GroupIndex'] = Unpaid_Jcodes['GroupIndex']
Unpaid_Jcodes_w_L['SubscriberIndex'] = Unpaid_Jcodes['SubscriberIndex']
Unpaid_Jcodes_w_L['SubgroupIndex'] = Unpaid_Jcodes['SubgroupIndex']
Unpaid_Jcodes_w_L['ClaimType'] = Unpaid_Jcodes['ClaimType']
Unpaid_Jcodes_w_L['ClaimSubscriberType'] = Unpaid_Jcodes['ClaimSubscriberType']
Unpaid_Jcodes_w_L['ClaimPrePrinceIndex'] = Unpaid_Jcodes['ClaimPrePrinceIndex']
Unpaid_Jcodes_w_L['ClaimCurrentStatus'] = Unpaid_Jcodes['ClaimCurrentStatus']
Unpaid_Jcodes_w_L['NetworkID'] = Unpaid_Jcodes['NetworkID']
Unpaid_Jcodes_w_L['AgreementID'] = Unpaid_Jcodes['AgreementID']

#And assign the target label 
Unpaid_Jcodes_w_L['IsUnpaid'] = 1
Unpaid_Jcodes_w_L_test['IsUnpaid'] = 1



#Look at the data..
print(Unpaid_Jcodes_w_L)
print(Unpaid_Jcodes_w_L_test)

# Do the same for the Paid set.


for v1 in Paid_Jcodes_test.dtype.names:
    Paid_Jcodes_w_L_test[v1] = Paid_Jcodes_test[v1]


#copy the data
Paid_Jcodes_w_L['V1'] = Paid_Jcodes['V1']
Paid_Jcodes_w_L['ClaimNumber'] = Paid_Jcodes['ClaimNumber']
Paid_Jcodes_w_L['ClaimLineNumber'] = Paid_Jcodes['ClaimLineNumber']
Paid_Jcodes_w_L['MemberID'] = Paid_Jcodes['MemberID']
Paid_Jcodes_w_L['ProviderID'] = Paid_Jcodes['ProviderID']
Paid_Jcodes_w_L['LineOfBusinessID'] = Paid_Jcodes['LineOfBusinessID']
Paid_Jcodes_w_L['RevenueCode'] = Paid_Jcodes['RevenueCode']
Paid_Jcodes_w_L['ServiceCode'] = Paid_Jcodes['ServiceCode']
Paid_Jcodes_w_L['PlaceOfServiceCode'] = Paid_Jcodes['PlaceOfServiceCode']
Paid_Jcodes_w_L['ProcedureCode'] = Paid_Jcodes['ProcedureCode']
Paid_Jcodes_w_L['DiagnosisCode'] = Paid_Jcodes['DiagnosisCode']
Paid_Jcodes_w_L['ClaimChargeAmount'] = Paid_Jcodes['ClaimChargeAmount']
Paid_Jcodes_w_L['DenialReasonCode'] = Paid_Jcodes['DenialReasonCode']
Paid_Jcodes_w_L['PriceIndex'] = Paid_Jcodes['PriceIndex']
Paid_Jcodes_w_L['InOutOfNetwork'] = Paid_Jcodes['InOutOfNetwork']
Paid_Jcodes_w_L['ReferenceIndex'] = Paid_Jcodes['ReferenceIndex']
Paid_Jcodes_w_L['PricingIndex'] = Paid_Jcodes['PricingIndex']
Paid_Jcodes_w_L['CapitationIndex'] = Paid_Jcodes['CapitationIndex']
Paid_Jcodes_w_L['SubscriberPaymentAmount'] = Paid_Jcodes['SubscriberPaymentAmount']
Paid_Jcodes_w_L['ProviderPaymentAmount'] = Paid_Jcodes['ProviderPaymentAmount']
Paid_Jcodes_w_L['GroupIndex'] = Paid_Jcodes['GroupIndex']
Paid_Jcodes_w_L['SubscriberIndex'] = Paid_Jcodes['SubscriberIndex']
Paid_Jcodes_w_L['SubgroupIndex'] = Paid_Jcodes['SubgroupIndex']
Paid_Jcodes_w_L['ClaimType'] = Paid_Jcodes['ClaimType']
Paid_Jcodes_w_L['ClaimSubscriberType'] = Paid_Jcodes['ClaimSubscriberType']
Paid_Jcodes_w_L['ClaimPrePrinceIndex'] = Paid_Jcodes['ClaimPrePrinceIndex']
Paid_Jcodes_w_L['ClaimCurrentStatus'] = Paid_Jcodes['ClaimCurrentStatus']
Paid_Jcodes_w_L['NetworkID'] = Paid_Jcodes['NetworkID']
Paid_Jcodes_w_L['AgreementID'] = Paid_Jcodes['AgreementID']

#And assign the target label 
Paid_Jcodes_w_L['IsUnpaid'] = 0
Paid_Jcodes_w_L_test['IsUnpaid'] = 0

#Look at the data..
print(Paid_Jcodes_w_L)
print(Paid_Jcodes_w_L_test)

#now combine the rows together (axis=0)
Jcodes_w_L = np.concatenate((Unpaid_Jcodes_w_L, Paid_Jcodes_w_L), axis=0)
Jcodes_w_L_test = np.concatenate((Unpaid_Jcodes_w_L_test, Paid_Jcodes_w_L_test), axis=0)
#check the shape
Jcodes_w_L.shape
Jcodes_w_L_test.shape

#44961 + 6068

#look at the transition between the rows around row 44961
print(Jcodes_w_L[44959:44964])
print(Jcodes_w_L_test[44959:44964])



ProviderID_IsUnpaid = rfn.merge_arrays((Jcodes_w_L_test['ProviderID'],Jcodes_w_L_test['IsUnpaid']), flatten = True, usemask = False)

#GroupBy JCodes using a dictionary
ProID_IsUnpaid_dict = {}
#Aggregate with Jcodes - code  modifiedfrom a former student's code, Anthony Schrams

for aJCode in ProviderID_IsUnpaid:
    #print(aJCode[0],type(aJCode[1]))
    if aJCode[0] in ProID_IsUnpaid_dict.keys():
        ProID_IsUnpaid_dict[aJCode[0]].append(int(aJCode[1]))
    else:
        ProID_IsUnpaid_dict[aJCode[0]] = [int(aJCode[1])]
        #print(aJCode[0])


markers = {"Lunch": "s", "Dinner": "X"} = '"FA0001389001"'.encode()
yy = '"FA0001411001"'.encode()

len(ProID_IsUnpaid_dict[vv])
sum(ProID_IsUnpaid_dict[vv])
len(ProID_IsUnpaid_dict[yy])
sum(ProID_IsUnpaid_dict[yy])

key_list, paid_list, unpaid_list, percentage_list = ([] for i in range(4))
for k, v in ProID_IsUnpaid_dict.items():
    print(k,len(v))
    key_list.append(k)
    paid_list.append(sum(v))
    unpaid_list.append(len(v)-sum(v))
    percentage_list.append((len(v)-sum(v))/len(v))

Paid_unPaid = [paid_list, unpaid_list]

import matplotlib.pyplot as plt
import seaborn as sns
#plt.plot(list(lr.keys()),list(lr.values()))
#plt.plot(list(lr.keys()),list(lr.values()))

#plt.scatter(paid_list,unpaid_list)
#sns.scatterplot(paid_list,percentage_list)

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

fig, ax = plt.subplots(figsize=(9,6))
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 11
ax = sns.scatterplot(x=paid_list, y=percentage_list,hue=key_list,palette=pkmn_type_colors,s=150)
#ax = sns.scatterplot(x=paid_list[9:14], y=percentage_list[9:14],hue=key_list[9:14],palette=pkmn_type_colors[9:14],s=150,marker='v')
plt.legend(bbox_to_anchor=(1, 1), loc=1)
ax.set_title("Percent Unpaid Claims vs Paid Claims")
ax.set(xlabel='Paid Claims', ylabel='Unpaid Claims %')
# add annotations one by one with a loop
#for line in range(0,len(paid_list)):
#     ax.text(paid_list[line]+225, percentage_list[line], key_list[line], horizontalalignment='left', size=10, color='black', weight='semibold')

sns.plt.show()

filename = 'PercentUnpaid_vs_Paid'
plt.savefig(filename, bbox_inches = 'tight')



#We need to shuffle the rows before using classifers in sklearn
Jcodes_w_L.dtype.names



#shuffle the rows
# Shuffle example:
        
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]


data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                          'formats':('U10', 'i4', 'f8')})
print(data.dtype)

data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)

#shuffle rows
np.random.shuffle(data)

#notice that the fields are still in the right order, but the rows have been shuffled.
print(data)

# We want to do the same for our data since we have combined unpaid and paid together, in that order. 

print(Jcodes_w_L[44957:44965])

# Apply the random shuffle
np.random.shuffle(Jcodes_w_L)


print(Jcodes_w_L[44957:44965])

#Columns are still in the right order
Jcodes_w_L

#Now get in the form for sklearn
Jcodes_w_L.dtype.names


# recall the features names:
#features = ['V1', 'ClaimNumber', 'ClaimLineNumber', 'MemberID', 'ProviderID',
#     'LineOfBusinessID', 'RevenueCode', 'ServiceCode', 'PlaceOfServiceCode',
#     'ProcedureCode', 'DiagnosisCode', 'ClaimChargeAmount', 'DenialReasonCode',
#     'PriceIndex', 'InOutOfNetwork', 'ReferenceIndex', 'PricingIndex',
#     'CapitationIndex', 'SubscriberPaymentAmount', 'ProviderPaymentAmount',
#     'GroupIndex', 'SubscriberIndex', 'SubgroupIndex', 'ClaimType',
#     'ClaimSubscriberType', 'ClaimPrePrinceIndex', 'ClaimCurrentStatus',
#     'NetworkID', 'AgreementID']

label =  'IsUnpaid'

#cat_features = ['V1', 'ProviderID','LineOfBusinessID','RevenueCode',
#                'ServiceCode', 'PlaceOfServiceCode', 'ProcedureCode',
#                'DiagnosisCode', 'DenialReasonCode',
#                'PriceIndex', 'InOutOfNetwork', 'ReferenceIndex', 
#                'PricingIndex', 'CapitationIndex', 'ClaimSubscriberType',
#                'ClaimPrePrinceIndex', 'ClaimCurrentStatus', 'NetworkID',
#                'AgreementID', 'ClaimType']

# Removed V1 and Diagnosis Code
cat_features = ['ProviderID','LineOfBusinessID','RevenueCode', 
                'ServiceCode', 'PlaceOfServiceCode', 'ProcedureCode',
                'DenialReasonCode','PriceIndex', 'InOutOfNetwork', 'ReferenceIndex', 
                'PricingIndex', 'CapitationIndex', 'ClaimSubscriberType',
                'ClaimPrePrinceIndex', 'ClaimCurrentStatus', 'NetworkID',
                'AgreementID', 'ClaimType']

numeric_features = ['ClaimNumber', 'ClaimLineNumber', 'MemberID', 
                    'ClaimChargeAmount',
                    'SubscriberPaymentAmount', 'ProviderPaymentAmount',
                    'GroupIndex', 'SubscriberIndex', 'SubgroupIndex']


#convert features to list, then to np.array 
# This step is important for sklearn to use the data from the structured NumPy array

#separate categorical and numeric features
Mcat = np.array(Jcodes_w_L[cat_features].tolist())
Mnum = np.array(Jcodes_w_L[numeric_features].tolist())

L = np.array(Jcodes_w_L[label].tolist())


######## LabelEncoder is not required for OneHotEncoder in current version of sklearn ########

# first use Sklearn's LabelEncoder function ... then use the OneHotEncoder function

# https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
# http://www.stephacking.com/encode-categorical-data-labelencoder-onehotencoder-python/

# Some claim you can do OnehotEncoder without a label encoder, but I haven't seen it work.
# https://stackoverflow.com/questions/48929124/scikit-learn-how-to-compose-labelencoder-and-onehotencoder-with-a-pipeline


# Run the Label encoder
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder.transform
#le = preprocessing.LabelEncoder()
#for i in range(18):
#   Mcat[:,i] = le.fit_transform(Mcat[:,i])

#for i in range(18):
#   Mcat[:,i] = le.transform(Mcat[:,i])



####################################################################################################


# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing


# Run the OneHotEncoder
# You can encounter a memory error here in which case, you probably should subset.
ohe = OneHotEncoder(sparse=False) #Easier to read
Mcat = ohe.fit_transform(Mcat)

#If you want to go back to the original mappings.
ohe.inverse_transform(Mcat)
ohe_features = ohe.get_feature_names(cat_features).tolist()

#What is the shape of the matrix categorical columns that were OneHotEncoded?   
Mcat.shape
Mnum.shape


#You can subset if you have memory issues.
#You might be able to decide which features are useful and remove some of them before the one hot encoding step

#If you want to recover from the memory error then subset
#Mcat = np.array(Jcodes_w_L[cat_features].tolist())

Mcat_subset = Mcat[0:10000]
Mcat_subset.shape

Mnum_subset = Mnum[0:10000]
Mnum_subset.shape

L_subset = L[0:10000]

# Uncomment if you need to run again from a subset.


#What is the size in megabytes before subsetting?
# https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-33.php
# and using base2 (binary conversion), https://www.gbmb.org/bytes-to-mb
print("%d Megabytes" % ((Mcat.size * Mcat.itemsize)/1048576))
print("%d Megabytes" % ((Mnum.size * Mnum.itemsize)/1048576))

#What is the size in megabytes after subsetting?
print("%d Megabytes" % ((Mcat_subset.size * Mcat_subset.itemsize)/1048576)) 
print("%d Megabytes" % ((Mnum_subset.size * Mnum_subset.itemsize)/1048576))

#Concatenate the columns
M = np.concatenate((Mcat, Mnum), axis=1)
#M = np.concatenate((Mcat_subset, Mnum_subset), axis=1)


L = Jcodes_w_L[label].astype(int)

# Match the label rows to the subset matrix rows.
#L = L[0:10000]

M.shape
L.shape

# Now you can use your DeathToGridsearch code.


