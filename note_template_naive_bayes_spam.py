# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 11:48:43 2016

@author: jiang_y
"""
# Ref: https://github.com/justmarkham/pycon-2016-tutorial


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# read file into pandas using a relative path
path = 'data/sms.tsv'
sms = pd.read_table(path, header=None, names=['label', 'message'])

# examine the first 10 rows
sms.head(10)
# examine the class distribution
sms.label.value_counts()
# convert label to a numerical variable
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})
# check that the conversion worked
sms.head(10)

# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
X = sms.message
y = sms.label_num

# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# instantiate the vectorizer
vect = CountVectorizer()

# learn training data vocabulary, then use it to create a document-term matrix
vect.fit(X_train)

# transform both training and testing data (using fitted vocabulary) into a document-term matrix
X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)

# convert sparse matrix to a dense matrix
X_train_dtm.toarray()


# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# train the model using X_train_dtm
nb.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)

# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# print message text for the false positives (ham incorrectly classified as spam)
X_test[(y_pred_class==1) & (y_test==0)]

# print message text for the false negatives (spam incorrectly classified as ham)
X_test[(y_pred_class==0) & (y_test==1)]

# example false negative
X_test[3132]

# calculate predicted probabilities for X_test_dtm (poorly calibrated)
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_pred_prob


# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)

# ## Part 7: Examining a model for further insight
# 
# We will examine the our **trained Naive Bayes model** to calculate the approximate **"spamminess" of each token**.

# store the vocabulary of X_train
X_train_tokens = vect.get_feature_names()
len(X_train_tokens)

# examine the first 50 tokens
print(X_train_tokens[0:50])
# examine the last 50 tokens
print(X_train_tokens[-50:])


# Naive Bayes counts the number of times each token appears in each class
nb.feature_count_

# rows represent classes, columns represent tokens
nb.feature_count_.shape


# number of times each token appears across all HAM messages
ham_token_count = nb.feature_count_[0, :]
ham_token_count


# number of times each token appears across all SPAM messages
spam_token_count = nb.feature_count_[1, :]
spam_token_count


# create a DataFrame of tokens with their separate ham and spam counts
tokens = pd.DataFrame({'token':X_train_tokens, 'ham':ham_token_count, 'spam':spam_token_count}).set_index('token')
tokens.head()


# examine 5 random DataFrame rows
tokens.sample(50)


# Naive Bayes counts the number of observations in each class
nb.class_count_

# Before we can calculate the "spamminess" of each token, we need to avoid **dividing by zero** and account for the **class imbalance**.

# add 1 to ham and spam counts to avoid dividing by 0
tokens['ham'] = tokens.ham + 1
tokens['spam'] = tokens.spam + 1
tokens.sample(5, random_state=6)


# convert the ham and spam counts into frequencies
tokens['ham'] = tokens.ham / nb.class_count_[0]
tokens['spam'] = tokens.spam / nb.class_count_[1]
tokens.sample(5, random_state=6)


# calculate the ratio of spam-to-ham for each token
tokens['spam_ratio'] = tokens.spam / tokens.ham
tokens.sample(5, random_state=6)


# examine the DataFrame sorted by spam_ratio
# note: use sort() instead of sort_values() for pandas 0.16.2 and earlier
tokens.sort_values('spam_ratio', ascending=False)



