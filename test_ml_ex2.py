# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:06:59 2016

@author: jiang_y
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cross_validation import train_test_split

myPath = 'C:/Users/jiang_y/Documents/MachineLearning/Coursera/machine-learning-ex2/ex2/'
fileName = 'ex2data1.txt'

df = pd.read_csv(myPath+fileName,header=None)
df.columns = ['x1','x2','type']

import seaborn as sns
%matplotlib inline
sns.pairplot(df, hue="type", vars=['x1','x2'],palette="husl")


X = df[['x1','x2']].as_matrix()
y = df['type'].as_matrix()

logreg = LogisticRegression()
logreg.fit(X,y)
y_pred = logreg.predict(X)
print("\tPrecision: %1.3f" % precision_score(y, y_pred))
print("\tRecall: %1.3f" % recall_score(y, y_pred))
print("\tF1: %1.3f\n" % f1_score(y, y_pred))

logreg.score(X,y)

data = pd.DataFrame()
for i in [0.01,0.03,0.1,0.3,1,3,10,30,100]:
    logreg = LogisticRegression(C=i)
    logreg.fit(X,y)
    y_pred = logreg.predict(X)
    print("\tC: %1.3f" % i)
    print("\tPrecision: %1.3f" % precision_score(y, y_pred))
    print("\tRecall: %1.3f" % recall_score(y, y_pred))
    print("\tF1: %1.3f\n" % f1_score(y, y_pred))
    temp = pd.DataFrame({'precision': [precision_score(y, y_pred)],
                  'recall': [recall_score(y, y_pred)],
                  'F1': [f1_score(y, y_pred)]})
    data = pd.concat([data,temp])
    
data = data.reset_index(drop=True)
data.plot()
