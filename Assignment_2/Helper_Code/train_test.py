# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 10:41:16 2019

@author: Mimir
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import dill as pickle #use "conda install dill" to install this in your conda env


X, y = make_classification(n_samples=60000, n_features=20, n_classes=2, n_redundant=5, n_informative=4,
                             n_clusters_per_class=3, shuffle=True, random_state = 2019)

#split into data sets
#80% train, 10% test, 10% validation (which simulates production)
#training size: .8 * 60000 = 48000
#test size: .1 * 60000 = 6000
#validation size: .1 * 60000 = 6000

#Originally take 80% and 20% split to get train and test
#training size: .8 * 60000 = 48000
#test size: .2 * 60000 = 12000
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Then take 50% of test set for validation data set.
#test size: .5 * 12000 = 6000
#validation size: .5 * 12000 = 6000
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)
 
#now we can write our x_val and y_val objects to pickles
with open('X_validation_values.pkl', 'wb') as file:
    pickle.dump(X_val, file)

with open('y_validation_values.pkl', 'wb') as file:
    pickle.dump(y_val, file)

   
#now scale
scaler = MinMaxScaler(feature_range=(0, 1))

#at this point, we need to save the original X-values 
# to later apply to the production/validation data
with open('X_train_values.pkl', 'wb') as file:
    pickle.dump(X_train, file)

#now fit and transform the scaler to training data
X_train = scaler.fit_transform(X_train) 

#now only transform X_test using the previously fit scaler on X_train
x_test = scaler.transform(X_test)

#At this point, we would impute missing values with the X_train values
#by fitting on the X_train values and transforming on the X_test values.
#Then, on the production/validation data set,  we would follow the same.
# We would fit on the X_train values and transform on the X_val values.

#Now we can create our random forest
clf = RandomForestClassifier(n_estimators=1000, max_depth=6, min_samples_split=4,
                             random_state=0)

#Now we can fit our random forest on the training data.
clf.fit(X_train, y_train)

#Now we predict on the test data
y_pred = clf.predict(X_test)

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print('\n')

#Now we dump our model to a pickled file.
with open('RF_classifier.pkl', 'wb') as file:
    pickle.dump(clf, file)
