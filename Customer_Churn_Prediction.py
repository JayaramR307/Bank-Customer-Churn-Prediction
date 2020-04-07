#!/usr/bin/env python
# coding: utf-8
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from statistics import mode
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('churn_prediction.csv')
print("\nFirst 5 rows: \n",data.head())

print("\nShape of the dataset:\n",data.shape)

print("\nMissing Values: ",data.isnull().sum())

# Imputing missing values
print("\nImputing Missing Values...")
data['gender'].fillna((data['gender'].mode()[0]), inplace=True)
data['dependents'].fillna((data['dependents'].mean()), inplace=True)
data['occupation'].fillna((data['occupation'].mode()[0]), inplace=True)
data['city'].fillna((data['city'].mode()[0]), inplace=True)
data['days_since_last_transaction'].fillna((data['days_since_last_transaction'].mean()), inplace=True)

# Standardizing

data = pd.get_dummies(data)


# Seperating independent and dependent variables
x = data.drop(['churn'], axis=1)
y = data['churn']

# Splitting Test and Train Data

train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 56, stratify=y)

# Applying Logistic Regression

model1 = LogisticRegression()
model1.fit(train_x,train_y)
pred1=model1.predict(test_x)
print("\nAccuracy of Logistic Regression:" ,model1.score(test_x, test_y))

# Applying KNN

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(train_x,train_y)
pred2=model2.predict(test_x)
print( "\nAccuracy of KNN:" ,model2.score(test_x, test_y))

# Applying Decision Tree Classifier 1

model3 = DecisionTreeClassifier(max_depth=3)
model3.fit(train_x,train_y)
pred3=model3.predict(test_x)
print( "\nAccuracy of Decision Tree Classifier 1:" ,model3.score(test_x, test_y))

# Applying Random Forest Classifier

model4 = RandomForestClassifier(n_estimators=100, max_depth= 4, random_state=3)
model4.fit(train_x, train_y)
pred4=model4.predict(test_x)
print( "\nAccuracy of Random Forest Classifier:" ,model4.score(test_x, test_y))


# Applying Decision Tree Classifier 2


model5 = DecisionTreeClassifier(max_depth=3, random_state=10)
model5.fit(train_x, train_y)
pred5=model5.predict(test_x)
print( "\nAccuracy of Decision Tree Classifier 2:" ,model5.score(test_x, test_y))

# Score of each models
# Here Decision Tree is used 2 times to remove Statistical Error

m1_score= model1.score(test_x, test_y)
m2_score= model2.score(test_x, test_y)
m3_score= model3.score(test_x, test_y)
m4_score= model4.score(test_x, test_y)
m5_score= model5.score(test_x, test_y)

final_pred = np.array([])
for i in range(0,len(test_x)):
    final_pred = np.append(final_pred,mode([pred1[i], pred2[i], pred3[i], pred4[i], pred5[i]]))

#Final Accuarcy

print("\nFinal Accuracy: ",accuracy_score(test_y, final_pred))