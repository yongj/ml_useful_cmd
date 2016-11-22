# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 08:28:39 2016

@author: jiang_y
"""
################################## Modules ####################################
# Important Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets, neighbors, linear_model

# Model package
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# calculate time
from datetime import datetime
start = datetime.now()
elapsed = datetime.now() - start
print(start)
print(elapsed)

################################### load data set ############################
import pandas as pd
df = pd.read_csv(myPath+fileName)

from sklearn import datasets
# Iris data set
iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
y = iris.target
# Digits data set
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Data operation
[m,n]=X.shape
m=len(X)

pd.DataFrame(X).describe()

# split data set for cross validation
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

################################# Training #####################################
# instantiate the model (using the default parameters)
logreg = LogisticRegression()
# fit the model with data
logreg.fit(X, y)
# predict the response for new observations
logreg.predict(X_new)

# Quick trainings
knn = neighbors.KNeighborsClassifier()
print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
logistic = linear_model.LogisticRegression()
print('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))

# scores
print("\tBrier: %1.3f" % (clf_score))
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))


######################### Plotting ###########################
%matplotlib qt          # plot in new window, use this in script: get_ipython().run_line_magic('matplotlib', 'qt'). Need to run "from IPython import get_ipython"
%matplotlib inline      # plot in ipython console

# Scatter plot
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xticks(())      # remove x ticks
plt.yticks(())      # remove y ticks

# pandas plotting tolls
import pandas as pd
X_df = pd.DataFrame(X)
colors_palette = {0: 'red', 1: 'yellow', 2:'blue'}
colors = [colors_palette[c] for c in y]
pd.tools.plotting.scatter_matrix(X_df,c=colors,diagonal='kde')     # 'kde','hist'
pd.tools.plotting.scatter_matrix(X_df,c=colors,diagonal='kde',alpha=0.2,figsize=(15, 15)) 
X_df.boxplot()
X_df.hist() 


import seaborn as sns
sns.jointplot(x="x", y="y", data=df); 
sns.pairplot(iris, hue="species", palette="husl")

