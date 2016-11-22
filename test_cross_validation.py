# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 08:02:00 2016

@author: jiang_y
"""

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# read in the iris data
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target


from sklearn.cross_validation import cross_val_score

# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())

# search for an optimal value of K for KNN
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

import matplotlib.pyplot as plt
%matplotlib inline

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# 10-fold cross-validation with the best KNN model
knn = KNeighborsClassifier(n_neighbors=20)
print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())

# 10-fold cross-validation with logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())


myPath = 'C:/Users/jiang_y/Documents/MachineLearning/Scripts/Data/Faults/'
fileName = 'faults.csv'
import pandas as pd

df = pd.read_csv(myPath+fileName,header=None)

import seaborn as sns
sns.pairplot(df.iloc[:,0:5])
color = pd.DataFrame(df.iloc[:,33])
color.iloc[:,0] = color.iloc[:,0]==1

df.iloc[:,33] = df.iloc[:,33].astype('object')

df1 = pd.concat([df.iloc[:,0:5], df.iloc[:,33]], axis=1)
df1.columns = ['X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Pixels_Areas','type7']
sns.pairplot(df1,hue='type7',palette="husl")



df['type'] = df.iloc[:,27] + df.iloc[:,28]*2 + df.iloc[:,29]*3 + df.iloc[:,30]*4 + df.iloc[:,31]*5 + df.iloc[:,32]*6 + df.iloc[:,33]*7
sns.pairplot(df,vars=[0,1,2,3],hue='type')
sns.pairplot(df,vars=[4,5,6,7],hue='type')
sns.pairplot(df,vars=[8,9,10,11],hue='type')
sns.pairplot(df,vars=[12,13,14,15],hue='type')
sns.pairplot(df,vars=[16,17,18,19],hue='type')
sns.pairplot(df,vars=[20,21,22,23],hue='type')
sns.pairplot(df,vars=[24,25,26],hue='type')
sns.pairplot(df,vars=[0,1,2,3,4,5],hue=27)

sns.pairplot(df.iloc[:,0:5])
sns.pairplot(df.iloc[:,0:5],hue=color,palette="husl")
sns.jointplot(x=2, y=3, data=df)

from pandas.tools.plotting import scatter_matrix
scatter_matrix(df.iloc[:,0:5], alpha=0.2, figsize=(6, 6), diagonal='kde') 