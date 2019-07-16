# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 11:53:37 2019

@author: Mimir
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import dill as pickle #use "conda install dill" to install this in your conda env
import lime #use 'conda install -c conda-forge lime" according to https://anaconda.org/conda-forge/lime
import lime.lime_tabular


#read in your X_train values and X and y validation values
with open('X_train_values.pkl', 'rb') as file:
    X_train = pickle.load(file)
    
with open('X_validation_values.pkl', 'rb') as file:
    X_val = pickle.load(file)

with open('y_validation_values.pkl', 'rb') as file:
    y_val = pickle.load(file)    

#now scale
scaler = MinMaxScaler(feature_range=(0, 1))

#now fit and transform the scaler to training data
X_train = scaler.fit_transform(X_train) 

#now only transform X_test using the previously fit scaler on X_train
X_val = scaler.transform(X_val)

#At this point, we would impute missing values with the X_train values
#by fitting on the X_train values and transforming on the X_test values.
#Then, on the production/validation data set,  we would follow the same.
# We would fit on the X_train values and transform on the X_val values.

#Now we load in our pickled model
with open('RF_classifier.pkl', 'rb') as file:
    clf = pickle.load(file)  

#Now we predict on the test data
y_pred = clf.predict(X_val)

#Report the Confusion Matrix and Classification Report results
# note: In a real production situation, we might not know the 
#       y_val values until later.

print("=== Confusion Matrix ===")
print(confusion_matrix(y_val, y_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_val, y_pred))
print('\n')

#get shape of X_vals for colnames
X_val.shape

#Match shape with feature names for explanations
X_colnames = ['feat1',
              'feat2',
              'feat3',
              'feat4',
              'feat5',
              'feat6',
              'feat7',
              'feat8',
              'feat9',
              'feat10',
              'feat11',
              'feat12',
              'feat13',
              'feat14',
              'feat15',
              'feat16',
              'feat17',
              'feat18',
              'feat19',
              'feat20']

#Note that our predict function first transforms the data into the one-hot representation
#encoder would be your "encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)"
#predict_fn = lambda x: clf.predict_proba(encoder.transform(x))

#https://eli5.readthedocs.io/en/latest/blackbox/lime.html
#https://buildmedia.readthedocs.org/media/pdf/lime-ml/latest/lime-ml.pdf

#create an explainer
np.random.seed(1)
explainer = lime.lime_tabular.LimeTabularExplainer(X_train ,class_names=['Not Strange', 'Strange, indeed.'], 
                                                   feature_names = X_colnames,
                                                   #If we have categorical features, use two lines, below:
                                                   #categorical_features=categorical_features, 
                                                   #categorical_names=categorical_names,
                                                   kernel_width=3, verbose=False)

#This is the validation row index being explained
i = 138
exp = explainer.explain_instance(X_val[i], clf.predict_proba, num_features=5)
#exp.show_in_notebook()
exp.save_to_file("pred_explaination.html")

#if I wanted to backtransform the X_val values to be better understood by the user
X_val_backtransformed = scaler.inverse_transform(X_val)

#Also consider looking at as_map() in Lime.
# you could get all the values it shows in the html file and write them to a CSV file,
# along with the prediction probabilities, prediction, and any identifier columns you need
# for cross-referencing.
