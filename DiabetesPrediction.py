# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:35:15 2022

@author: DELL
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



diabetes_dataset = pd.read_csv('diabetes.csv')

x = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
y = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 2)

model = svm.SVC(kernel = 'linear')

model.fit(x_train, y_train)

x_train_predicted = model.predict(x_train)
x_train_accuracy = accuracy_score(x_train_predicted, y_train)

x_test_predicted = model.predict(x_test)
x_test_accuracy = accuracy_score(x_test_predicted, y_test)





input_data = (5,116,74,0,0,25.6,0.201,30)
data_as_array = np.asarray(input_data)

data_reshaped = data_as_array.reshape(1, -1)

input_data = scaler.transform(data_reshaped)
prediction = model.predict(input_data)




