"""Import libraries"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""Importing data"""

dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
X_train = dataset_train.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1).values
y_train = dataset_train.loc[:, 'Survived'].values
X_test = dataset_test.drop(['Name', 'Ticket', 'Cabin'], axis=1).values
print(X_train[1].size)
print(y_train)
print(X_test[1].size)

"""Preparing data"""

from sklearn.compose import  ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2, -1])], remainder='passthrough')
X_train = np.array(ct.fit_transform(X_train))
print(X_train[1])
X_test = np.array(ct.transform(X_test))
print(X_test[2])

"""Training model"""

from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators =500, random_state = 42)
regressor.fit(X_train, y_train)

"""Predict data"""

y_pred = regressor.predict(X_test)
print(y_pred)
dataf = pd.DataFrame({'PassengerId': dataset_test['PassengerId'], 'Survived': y_pred})
dataf.to_csv('submission.csv', index=False)

"""Evaluate Performance"""

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

y_train_pred = regressor.predict(X_train)
cm = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix on training data:")
print(cm)
accuracy = accuracy_score(y_train, y_train_pred)
print(f"Accuracy Score on training data: {accuracy}")


accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
print("Cross-Validation Accuracy Scores (10 folds):")
print(accuracies)
print(f"Mean Accuracy: {accuracies.mean()*100:.2f} %")
print(f"Standard Deviation of Accuracy: {accuracies.std()*100:.2f} %")