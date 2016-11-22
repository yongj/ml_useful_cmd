# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 07:55:40 2016

@author: jiang_y
"""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
%matplotlib inline

# read in the iris data
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target


# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')

# search for an optimal value of K for KNN
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

from sklearn.grid_search import GridSearchCV

param_grid = {'n_neighbors':range(1,31)}
# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')

# fit the grid with data
grid.fit(X, y)


# view the complete results (list of named tuples)
grid.grid_scores_



weight_options = ['uniform', 'distance']
param_grid = {'n_neighbors':range(1,31),'weights':weight_options}              
# instantiate and fit the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X, y)