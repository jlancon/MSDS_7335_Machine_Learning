# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:45:07 2019

@author: Prodigy
"""

data = dict(a={'aa':3}, b={'aa':4}, c={'aa':7})
print(data)
for k1,v1 in data.items():
    print(v1)
    data_sorted = {k: v for k, v in sorted(v1.items(), key=lambda x: x[1])}

    for k2,v2 in v1.items():
        print(k2,v2,sum(v2))
        data_sorted = sorted(sum(data.items()), key=lambda x: x[1], reverse=True)
print(data_sorted)






for k1, v1 in clfDict.items():  # go through first level of clfDict
            if k1 in clfString:		# if clfString1 matches first level
                for k2,v2 in v1.items(): # go through the inner dictionary of hyper parameters
                    print(k2)			 # for each hyper parameter in the inner list..	
                    for vals in v2:		 # go through the values for each hyper parameter 
                        print(vals)		 # and show them...
                        
                        #pdb.set_trace()
                        
                        
                        
data = dict(a=1, b=3, c=2)
print(data)
data_sorted = {k: v for k, v in sorted(data.items(), key=lambda x: x[1])}
print(data_sorted)