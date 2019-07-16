# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 17:31:47 2019

@author: Chris
"""
#import libraries
import numpy as np
import numpy.lib.recfunctions as rfn
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
## A test NumPy array of type string
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
SCHEME = ('"j'.encode(),'"J'.encode(),'"b'.encode()) #You can use any number of combinations to seach
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
#print(InNetwork)
#print(InNetwork.dtype.names)

print('\n\n*** In Network J-Code Payments ***\n')
print('Number of In-Network JCode Procdures: ',len(InNetwork))

InNetworkPayments_JCodes = np.sum(InNetwork['ProviderPaymentAmount'])
print('Total Payments for In-Network J-Code Procedures:  $', round(InNetworkPayments_JCodes,2))

##############################################################

#Sorted Jcodes, by ProviderPaymentAmount
#Sorted_Jcodes = np.sort(Jcodes, order='ProviderPaymentAmount')

# Sorted Jcodes, by ProviderPaymentAmount; in decending order in 1 step
Sorted_Jcodes = np.sort(Jcodes, order='ProviderPaymentAmount')[::-1]

# Reverse the sorted Jcodes (A.K.A. in descending order)
#Sorted_Jcodes = Sorted_Jcodes[::-1]
# [7, 6, 5, 4, 3, 2, 1]


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

# We want to find the top five J-codes based on the payment to providers?
#Join and flatten subset of Sorted_Jcodes 'ProcedureCode' and 'ProviderPaymentAmount' 
Jcodes_with_ProviderPayments = rfn.merge_arrays((Sorted_Jcodes['ProcedureCode'],Sorted_Jcodes['ProviderPaymentAmount']), flatten = True, usemask = False)

# What does the result look like?
#print(Jcodes_with_ProviderPayments[:3])
#Jcodes_with_ProviderPayments.shape


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
    
#print the results        
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

#print(Sorted_Jcodes.dtype.names)

##We need to come up with labels for paid and unpaid Jcodes

## find unpaid row indexes  
#unpaid_mask = (Sorted_Jcodes['ProviderPaymentAmount'] == 0)

## find paid row indexes
#paid_mask = (Sorted_Jcodes['ProviderPaymentAmount'] > 0)


#Here are our
#Unpaid_Jcodes = Sorted_Jcodes[unpaid_mask]
#len(Unpaid_Jcodes)

#Paid_Jcodes = Sorted_Jcodes[paid_mask]
#len(Paid_Jcodes)

###### *******
Unpaid_Jcodes = Sorted_Jcodes[Sorted_Jcodes['ProviderPaymentAmount'] == 0]
Paid_Jcodes = Sorted_Jcodes[Sorted_Jcodes['ProviderPaymentAmount']  > 0]
#len(Unpaid_Jcodes)
#len(Paid_Jcodes)

print('\n\n*** Claims - J-Code Payments ***\n')
print('Total number of "PAID" J-Code Procedure claim lines:        ',len(Paid_Jcodes))
print('Total number of "UN-PAID" J-Code Procedure claim lines:     ',len(Unpaid_Jcodes))
print('Total percentage of "UN-PAID" J-Code Procedure claim lines: ',round((len(Unpaid_Jcodes)/len(Jcodes))*100,2),'%')


#These are still structured numpy arrays
#print(Unpaid_Jcodes.dtype.names)
#print(Unpaid_Jcodes[0])

#print(Paid_Jcodes.dtype.names)
#print(Paid_Jcodes[0])

# Need to create labels
# Create a new column and data type for both structured arrays
new_dtype = np.dtype(Unpaid_Jcodes.dtype.descr + [('IsUnpaid', '<i4')])
#new_dtype2 = np.dtype(Paid_Jcodes.dtype.descr + [('IsUnpaid', '<i4')])

#print(new_dtype)
#print(new_dtype2)

#create new structured array with labels

#first get the right shape for each.
Unpaid_Jcodes_w_L = np.zeros(Unpaid_Jcodes.shape, dtype=new_dtype)
Paid_Jcodes_w_L = np.zeros(Paid_Jcodes.shape, dtype=new_dtype)


#check the shape
#Unpaid_Jcodes_w_L.shape
#Paid_Jcodes_w_L.shape

#Look at the data
#print(Unpaid_Jcodes_w_L)
#print(Paid_Jcodes_w_L)


for v1 in Unpaid_Jcodes.dtype.names:
    Unpaid_Jcodes_w_L[v1] = Unpaid_Jcodes[v1]


#copy the data
#Unpaid_Jcodes_w_L['V1'] = Unpaid_Jcodes['V1']
#Unpaid_Jcodes_w_L['ClaimNumber'] = Unpaid_Jcodes['ClaimNumber']
#Unpaid_Jcodes_w_L['ClaimLineNumber'] = Unpaid_Jcodes['ClaimLineNumber']
#Unpaid_Jcodes_w_L['MemberID'] = Unpaid_Jcodes['MemberID']
#Unpaid_Jcodes_w_L['ProviderID'] = Unpaid_Jcodes['ProviderID']
#Unpaid_Jcodes_w_L['LineOfBusinessID'] = Unpaid_Jcodes['LineOfBusinessID']
#Unpaid_Jcodes_w_L['RevenueCode'] = Unpaid_Jcodes['RevenueCode']
#Unpaid_Jcodes_w_L['ServiceCode'] = Unpaid_Jcodes['ServiceCode']
#Unpaid_Jcodes_w_L['PlaceOfServiceCode'] = Unpaid_Jcodes['PlaceOfServiceCode']
#Unpaid_Jcodes_w_L['ProcedureCode'] = Unpaid_Jcodes['ProcedureCode']
#Unpaid_Jcodes_w_L['DiagnosisCode'] = Unpaid_Jcodes['DiagnosisCode']
#Unpaid_Jcodes_w_L['ClaimChargeAmount'] = Unpaid_Jcodes['ClaimChargeAmount']
#Unpaid_Jcodes_w_L['DenialReasonCode'] = Unpaid_Jcodes['DenialReasonCode']
#Unpaid_Jcodes_w_L['PriceIndex'] = Unpaid_Jcodes['PriceIndex']
#Unpaid_Jcodes_w_L['InOutOfNetwork'] = Unpaid_Jcodes['InOutOfNetwork']
#Unpaid_Jcodes_w_L['ReferenceIndex'] = Unpaid_Jcodes['ReferenceIndex']
#Unpaid_Jcodes_w_L['PricingIndex'] = Unpaid_Jcodes['PricingIndex']
#Unpaid_Jcodes_w_L['CapitationIndex'] = Unpaid_Jcodes['CapitationIndex']
#Unpaid_Jcodes_w_L['SubscriberPaymentAmount'] = Unpaid_Jcodes['SubscriberPaymentAmount']
#Unpaid_Jcodes_w_L['ProviderPaymentAmount'] = Unpaid_Jcodes['ProviderPaymentAmount']
#Unpaid_Jcodes_w_L['GroupIndex'] = Unpaid_Jcodes['GroupIndex']
#Unpaid_Jcodes_w_L['SubscriberIndex'] = Unpaid_Jcodes['SubscriberIndex']
#Unpaid_Jcodes_w_L['SubgroupIndex'] = Unpaid_Jcodes['SubgroupIndex']
#Unpaid_Jcodes_w_L['ClaimType'] = Unpaid_Jcodes['ClaimType']
#Unpaid_Jcodes_w_L['ClaimSubscriberType'] = Unpaid_Jcodes['ClaimSubscriberType']
#Unpaid_Jcodes_w_L['ClaimPrePrinceIndex'] = Unpaid_Jcodes['ClaimPrePrinceIndex']
#Unpaid_Jcodes_w_L['ClaimCurrentStatus'] = Unpaid_Jcodes['ClaimCurrentStatus']
#Unpaid_Jcodes_w_L['NetworkID'] = Unpaid_Jcodes['NetworkID']
#Unpaid_Jcodes_w_L['AgreementID'] = Unpaid_Jcodes['AgreementID']

#And assign the target label 
Unpaid_Jcodes_w_L['IsUnpaid'] = 1



#Look at the data..
#print(Unpaid_Jcodes_w_L)


# Do the same for the Paid set.

for v1 in Paid_Jcodes.dtype.names:
    Paid_Jcodes_w_L[v1] = Paid_Jcodes[v1]

#copy the data
#Paid_Jcodes_w_L['V1'] = Paid_Jcodes['V1']
#Paid_Jcodes_w_L['ClaimNumber'] = Paid_Jcodes['ClaimNumber']
#Paid_Jcodes_w_L['ClaimLineNumber'] = Paid_Jcodes['ClaimLineNumber']
#Paid_Jcodes_w_L['MemberID'] = Paid_Jcodes['MemberID']
#Paid_Jcodes_w_L['ProviderID'] = Paid_Jcodes['ProviderID']
#Paid_Jcodes_w_L['LineOfBusinessID'] = Paid_Jcodes['LineOfBusinessID']
#Paid_Jcodes_w_L['RevenueCode'] = Paid_Jcodes['RevenueCode']
#Paid_Jcodes_w_L['ServiceCode'] = Paid_Jcodes['ServiceCode']
#Paid_Jcodes_w_L['PlaceOfServiceCode'] = Paid_Jcodes['PlaceOfServiceCode']
#Paid_Jcodes_w_L['ProcedureCode'] = Paid_Jcodes['ProcedureCode']
#Paid_Jcodes_w_L['DiagnosisCode'] = Paid_Jcodes['DiagnosisCode']
#Paid_Jcodes_w_L['ClaimChargeAmount'] = Paid_Jcodes['ClaimChargeAmount']
#Paid_Jcodes_w_L['DenialReasonCode'] = Paid_Jcodes['DenialReasonCode']
#Paid_Jcodes_w_L['PriceIndex'] = Paid_Jcodes['PriceIndex']
#Paid_Jcodes_w_L['InOutOfNetwork'] = Paid_Jcodes['InOutOfNetwork']
#Paid_Jcodes_w_L['ReferenceIndex'] = Paid_Jcodes['ReferenceIndex']
#Paid_Jcodes_w_L['PricingIndex'] = Paid_Jcodes['PricingIndex']
#Paid_Jcodes_w_L['CapitationIndex'] = Paid_Jcodes['CapitationIndex']
#Paid_Jcodes_w_L['SubscriberPaymentAmount'] = Paid_Jcodes['SubscriberPaymentAmount']
#Paid_Jcodes_w_L['ProviderPaymentAmount'] = Paid_Jcodes['ProviderPaymentAmount']
#Paid_Jcodes_w_L['GroupIndex'] = Paid_Jcodes['GroupIndex']
#Paid_Jcodes_w_L['SubscriberIndex'] = Paid_Jcodes['SubscriberIndex']
#Paid_Jcodes_w_L['SubgroupIndex'] = Paid_Jcodes['SubgroupIndex']
#Paid_Jcodes_w_L['ClaimType'] = Paid_Jcodes['ClaimType']
#Paid_Jcodes_w_L['ClaimSubscriberType'] = Paid_Jcodes['ClaimSubscriberType']
#Paid_Jcodes_w_L['ClaimPrePrinceIndex'] = Paid_Jcodes['ClaimPrePrinceIndex']
#Paid_Jcodes_w_L['ClaimCurrentStatus'] = Paid_Jcodes['ClaimCurrentStatus']
#Paid_Jcodes_w_L['NetworkID'] = Paid_Jcodes['NetworkID']
#Paid_Jcodes_w_L['AgreementID'] = Paid_Jcodes['AgreementID']

#And assign the target label 
Paid_Jcodes_w_L['IsUnpaid'] = 0

#Look at the data..
#print(Paid_Jcodes_w_L)

#now combine the rows together (axis=0)
Jcodes_w_L = np.concatenate((Unpaid_Jcodes_w_L, Paid_Jcodes_w_L), axis=0)
#check the shape
Jcodes_w_L.shape

#44961(unpaid) + 6068(paid)

#look at the transition between the rows around row 44961
#print(Jcodes_w_L[44959:44964])



ProviderID_IsUnpaid = rfn.merge_arrays((Jcodes_w_L['ProviderID'],Jcodes_w_L['IsUnpaid']), flatten = True, usemask = False)

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


#vv = '"FA0001389001"'.encode()
#yy = '"FA0001411001"'.encode()
#len(ProID_IsUnpaid_dict[vv])
#sum(ProID_IsUnpaid_dict[vv])
#len(ProID_IsUnpaid_dict[yy])
#sum(ProID_IsUnpaid_dict[yy])

key_list, paid_list, unpaid_list, unpaid_percentage_list,totalClaims = ([] for i in range(5))
for k, v in ProID_IsUnpaid_dict.items():
    #print(k,len(v))
    key_list.append(k)
    unpaid_list.append(sum(v))
    paid_list.append(len(v)-sum(v))
    unpaid_percentage_list.append(round((((sum(v))/len(v))*100),3))
    totalClaims.append(len(v))


#Paid_unPaid = [paid_list, unpaid_list]
#sum(Paid_unPaid[0])
#sum(Paid_unPaid[1])
#percentunpaid = []
#for u in range(15):
#    percentunpaid[u] = (Paid_unPaid[1][u]/(Paid_unPaid[1][u]+Paid_unPaid[0][u]))

#plt.plot(list(lr.keys()),list(lr.values()))
#plt.plot(list(lr.keys()),list(lr.values()))

# A. Create a scatter plot that displays the number of unpaid claims
# (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each
# provider versus the number of paid claims.

#Creating color scheme for plot
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
#ax = sns.scatterplot(x=paid_list, y=unpaid_percentage_list,hue=key_list,palette=pkmn_type_colors,s=150)
ax = sns.scatterplot(x=paid_list, y=unpaid_list,hue=key_list,palette=pkmn_type_colors,s=150)
#ax = sns.scatterplot(x=paid_list[9:14], y=percentage_list[9:14],hue=key_list[9:14],palette=pkmn_type_colors[9:14],s=150,marker='v')
ax.legend(bbox_to_anchor=(1,0.65), loc=1)
ax.set_title("Fig.1 - Paid Claims vs Un-Paid Claims")
ax.set(xlabel='Paid Claims', ylabel='Unpaid Claims')
# add annotations one by one with a loop
#for line in range(0,len(paid_list)):
#     ax.text(paid_list[line]+225, percentage_list[line], key_list[line], horizontalalignment='left', size=8, color='black', weight='semibold')

plt.show()

filename = 'Fig_1_Unpaid_vs_Paid'
plt.savefig(filename, bbox_inches = 'tight')


########################################################################################
###### Creation of Fig 2 - Plot % Un-Paid J-Code Procedures vs Overall J-Code Procedures
fig, ax = plt.subplots(figsize=(9,7))
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 11
ax = sns.scatterplot(x=totalClaims, y=unpaid_percentage_list,hue=key_list,palette=pkmn_type_colors,s=150)
#ax = sns.scatterplot(x=paid_list[9:14], y=percentage_list[9:14],hue=key_list[9:14],palette=pkmn_type_colors[9:14],s=150,marker='v')
ax.legend(bbox_to_anchor=(1,0.65), loc=1)
ax.set_title("Fig.2 - Percent Unpaid Claims vs Total Claims")
ax.set(xlabel='Total Claims', ylabel='Unpaid Claims as % of Total Claims')
# add annotations one by one with a loop
#for line in range(0,len(paid_list)):
#     ax.text(paid_list[line]+225, percentage_list[line], key_list[line], horizontalalignment='left', size=8, color='black', weight='semibold')

plt.show()

filename = 'Fig_2_PercentUnpaid_vs_Paid'
plt.savefig(filename, bbox_inches = 'tight')


'''
2. For the following exercises, determine the number of providers that were paid for at least one J-code. Use the J-code claims for these providers to complete the following exercises.

   B1. Insights can you suggest from the graph (Fig 1)?
       i)   There are 51029 Procedural Codes that begin with 'J'.  Of that, 44961 (88.1%) are un-paid.
                Investigation as to the reason for this high number of Un-Paid J-Code Procedure claims
                should be initiated.
       ii) A few providers have an extreme number of unpaid claims(>8000):
                a) FA0001389001 Total Unpaid Claims: 14842
                b) FA0001387002 Total Unpaid Claims: 11585
                c) FA0001387001 Total Unpaid Claims: 8784
        An investigation into why these providers have so many unpaid claims should be initiated (i.e. improper
        paperwork, procedures, etc.)
    
    C. Based on the graph, is the behavior of any of the providers concerning? Explain.    
    C1. A second graphic was created, looking at the % of unpaid claims vs overall claim
        Based on the percentage graph (Fig 2), is the behavior of any of the providers concerning? Explain.
       i)   The majority of the providers have an average of over 80% un-paid J-Code claims, with 
            the exception of 3 providers:
                    a)  FA1000015001, 61.26%,  Total Claims: 1910 
                    b)  FA1000014001, 51.72%   Total Claims: 1162
                    c)  FA0004551001, 43.69%   Total Claims: 737
            The overall volume of un-paid claims made up by these 3 providers is not exorbidant
            but we should look into reasons for the high-un-paid claim percentage for these providers.
'''






'''
3. Consider all claim lines with a J-code.

     A. What percentage of J-code claim lines were unpaid?

     B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.

     C. How accurate is your model at predicting unpaid claims?

      D. What data attributes are predominately influencing the rate of non-payment?

'''

########################################################
########################################################
########################################################

#Prep for machine learning with classifiers in sklearn

########################################################
#We need to shuffle the rows before using classifers in sklearn
#Jcodes_w_L.dtype.names



#shuffle the rows
# Shuffle example:
        
#name = ['Alice', 'Bob', 'Cathy', 'Doug']
#age = [25, 45, 37, 19]
#weight = [55.0, 85.5, 68.0, 61.5]


#data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
#                          'formats':('U10', 'i4', 'f8')})
#print(data.dtype)

#data['name'] = name
#data['age'] = age
#data['weight'] = weight
#print(data)

#shuffle rows
#np.random.shuffle(data)

#notice that the fields are still in the right order, but the rows have been shuffled.
#print(data)

# We want to do the same for our data since we have combined unpaid and paid together, in that order. 
# Displaying observations that overlap the merger location of data set
print(Jcodes_w_L[44957:44965])



# Apply the random shuffle
np.random.RandomState(seed=244)
np.random.shuffle(Jcodes_w_L)


print(Jcodes_w_L[44957:44965])

#Columns are still in the right order
Jcodes_w_L

#Now get in the form for sklearn
#Jcodes_w_L.dtype.names


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
# Removed 'ClaimCurrentStatus':   Possible Data Leakage
cat_features = ['ProviderID','LineOfBusinessID','RevenueCode', 
                'ServiceCode', 'PlaceOfServiceCode', 'ProcedureCode',
                'DenialReasonCode','PriceIndex', 'InOutOfNetwork', 'ReferenceIndex', 
                'PricingIndex', 'CapitationIndex', 'ClaimSubscriberType',
                'ClaimPrePrinceIndex', 'NetworkID',
                'AgreementID', 'ClaimType']

# Removed 'ProviderPaymentAmount' since this variable is data leakage
numeric_features = ['ClaimNumber', 'ClaimLineNumber', 'MemberID', 
                    'ClaimChargeAmount',
                    'SubscriberPaymentAmount',
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

from sklearn.preprocessing import OneHotEncoder
# Run the OneHotEncoder
# You can encounter a memory error here in which case, you probably should subset.
ohe = OneHotEncoder(sparse=False) #Easier to read
Mcat = ohe.fit_transform(Mcat)

#If you want to go back to the original mappings.
#ohe.inverse_transform(Mcat)
#ohe_features = ohe.get_feature_names(cat_features).tolist()

#What is the shape of the matrix categorical columns that were OneHotEncoded?   
Mcat.shape
Mnum.shape


#You can subset if you have memory issues.
#You might be able to decide which features are useful and remove some of them before the one hot encoding step

#If you want to recover from the memory error then subset
#Mcat = np.array(Jcodes_w_L[cat_features].tolist())

#Mcat_subset = Mcat[0:10000]
#Mcat_subset.shape

#Mnum_subset = Mnum[0:10000]
#Mnum_subset.shape

#L_subset = L[0:10000]

# Uncomment if you need to run again from a subset.


#What is the size in megabytes before subsetting?
# https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-33.php
# and using base2 (binary conversion), https://www.gbmb.org/bytes-to-mb
print("%d Megabytes" % ((Mcat.size * Mcat.itemsize)/1048576))
print("%d Megabytes" % ((Mnum.size * Mnum.itemsize)/1048576))

#What is the size in megabytes after subsetting?
#print("%d Megabytes" % ((Mcat_subset.size * Mcat_subset.itemsize)/1048576)) 
#print("%d Megabytes" % ((Mnum_subset.size * Mnum_subset.itemsize)/1048576))

#Concatenate the columns
M = np.concatenate((Mcat, Mnum), axis=1)
#M = np.concatenate((Mcat_subset, Mnum_subset), axis=1)


L = Jcodes_w_L[label].astype(int)

# Match the label rows to the subset matrix rows.
#M = M[0:1000]
#L = L[0:1000]

M.shape
L.shape

# Now you can use your DeathToGridsearch code.




from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold 
from itertools import product
import statistics
import json
from warnings import simplefilter



# Part 1: Loading Data set - Breast Cancer Data Set within sklearn
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer
#cancer = datasets.load_breast_cancer()

# array M includes the X's/Matrix/the data
#M = cancer.data
#M.shape

# Array L includes Y values/labels/target
#L = cancer.target
#L.shape[0]


# Enter: Number of folds (k-fold) cross validation
n_folds = 10

# Creating List of Classifiers to use for analysis
clfsList = [RandomForestClassifier, KNeighborsClassifier] 

# Enter: Range of Hyper-parameters  user wishes to manipulate
# for each classifier listed
# NOTE: No effort was placed on improving Accuracy by manipulating hyper-parameters
# Paramters were chosen as examples only.
clfDict = {'RandomForestClassifier': {
                "min_samples_split": [2,3],
                "n_jobs": [1,2]},'KNeighborsClassifier': {
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
    print("\nClassifier with Paramters:",i[0],'\nk-Fold F-1 Scores',i[1],'\nMedian F1',statistics.median(i[1])) 


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
plt.rcParams["axes.titlesize"] = 35
plt.rcParams["axes.labelsize"] = 30
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
    ax.text(F1_median[tick]+0.001,pos[tick], median_labels[tick],
            verticalalignment='center', size='medium', color='b', weight='semibold',rotation=-90)
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0)

filename = 'clf_HW2F1_Boxplots'
plt.savefig(filename, bbox_inches = 'tight') # save the plot to the user's working directory

#_____________________ Box Plot - Horizontal Orientation - End ________________


