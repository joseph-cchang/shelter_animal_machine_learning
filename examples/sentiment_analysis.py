#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Use the sample_submission.csv dataset for the data. This code will implement SVM classifer, specifically linear kernel on the data.

# Import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
# Replace 'data.csv' with your actual dataset path
data = pd.read_csv('sample_submission.csv')

# Based on the columns of the data: ID,	Adoption,	Died,	Euthanasia,	Return_to_owner,	Transfer
# We will assume 'Adoption' is the target variable, and the rest are features
# We will drop 'ID' as it doesn't provide useful predictive information
X = data.drop(columns=['ID', 'Adoption'])
y = data['Adoption']

# Split the data into training and testing sets into 7/3 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Here, we will standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Here, we will create and train an SVM classifier. It will be Linear kernel
clf = SVC(kernel='linear', random_state=123)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model on accuracy and classification report
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

