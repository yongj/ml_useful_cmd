# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 11:39:24 2016

@author: jiang_y
"""
# Ref: https://github.com/justmarkham/pycon-2016-tutorial

import pandas as pd

# example text for model training (SMS messages)
simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']

# import and instantiate CountVectorizer (with the default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()


# learn the 'vocabulary' of the training data (occurs in-place)
vect.fit(simple_train)


# examine the fitted vocabulary
vect.get_feature_names()


# transform training data into a 'document-term matrix'
simple_train_dtm = vect.transform(simple_train)
simple_train_dtm


# convert sparse matrix to a dense matrix
simple_train_dtm.toarray()


# examine the vocabulary and document-term matrix together
pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
# check the type of the document-term matrix
type(simple_train_dtm)


# examine the sparse matrix contents
print(simple_train_dtm)

# example text for model testing
simple_test = ["please don't call me"]


# In order to **make a prediction**, the new observation must have the **same features as the training observations**, both in number and meaning.

# transform testing data into a document-term matrix (using existing vocabulary)
simple_test_dtm = vect.transform(simple_test)
simple_test_dtm.toarray()


# examine the vocabulary and document-term matrix together
pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names())
