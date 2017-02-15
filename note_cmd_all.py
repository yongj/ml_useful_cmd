# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 08:28:39 2016

@author: jiang_y
"""
python --version # need to issue in cmd window

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

################################## System Operations ####################################
# calculate time
from datetime import datetime
start = datetime.now()
elapsed = datetime.now() - start
print(start)
print(elapsed)

% time      # ipython syntax to log run time

# generate random number 0~9
from random import randrange
print(randrange(0,10))  

# directory operations
import os
os.getcwd()
os.chdir(r'C:\Users\jiang_y\Documents\MachineLearning\spyder')
os.listdir()
os.path.join()

# get file list
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
data_dir = '/tmp/cifar10_data'
filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]

################################### load data set ############################
import pandas as pd
df = pd.read_csv(myPath+fileName)

url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
sms = pd.read_table(url, header=None, names=['label', 'message'])

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
X = data.reshape((m,n))

data = data.reset_index(inplace=False) # reset index. The inplace flag = 1 will maintian original index

# convert label to a numerical variable
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})

# split data set for cross validation
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)


# dump to and load from pickle
from six.moves import cPickle as pickle
pickle_file = 'notMNIST.pickle'
try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise 

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  del save  # hint to help gc free up memory  

# print an image. ref: http://matplotlib.org/users/image_tutorial.html
import matplotlib.pyplot as plt
img=mpimg.imread(filePath)
imgplot = plt.imshow(img)       # image dimensions are: HEIGHT * WIDTH * DEPTH (a.k.a RGB channels)
  
################################# Training #####################################
# instantiate the model (using the default parameters)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg = LogisticRegression(multi_class = 'multinomial', solver='sag')  # multinomial logistic regression (softmax)
# fit the model with data
logreg.fit(X, y)
# predict the response for new observations
logreg.predict(X_new)

# Quick trainings
knn = neighbors.KNeighborsClassifier()
print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
logistic = linear_model.LogisticRegression()
print('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))

# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
# train the model using X_train_dtm (timing it with an IPython "magic command")
%time nb.fit(X_train_dtm, y_train)

################################# Evaluation #####################################
# scores
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)
metrics.precision_score(y_test, y_pred)
metrics.recall_score(y_test, y_pred)
metrics.f1_score(y_test, y_pred)
metrics.confusion_matrix(y_test, y_pred)      # print the confusion matrix

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

