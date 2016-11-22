# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:32:25 2016

@author: jiang_y
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cross_validation import train_test_split

myPath = 'C:/Users/jiang_y/Documents/MachineLearning/Scripts/Data/Faults/'
fileName = 'faults.csv'

df = pd.read_csv(myPath+fileName,header=None)
temp = df.as_matrix()
X = temp[:,0:27]
y = temp[:,27]
y = temp[:,27] + temp[:,28]*2 + temp[:,29]*3 + temp[:,30]*4 + temp[:,31]*5 + temp[:,32]*6 + temp[:,33]*7

y = y.astype('int')

logreg = LogisticRegression()
score = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
print(score.mean())

logreg = LogisticRegression(verbose=True)
logreg.fit(X,y)
y_pred = logreg.predict(X)
print("\tPrecision: %1.3f" % precision_score(y, y_pred))
print("\tRecall: %1.3f" % recall_score(y, y_pred))
print("\tF1: %1.3f\n" % f1_score(y, y_pred))

for i in range(7):
    y = temp[:,27+i]
    logreg = LogisticRegression()
    score = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
    print(score.mean())


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
y_train = y_train.astype('int')
logreg = LogisticRegression(C=1)
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print(max(y_pred))
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))