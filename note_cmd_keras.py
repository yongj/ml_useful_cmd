# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 08:31:52 2017

@author: jiang_y
"""

val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
filenames = val_batches.filenames
val_classes = val_batches.classes

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D


model = Sequential()

model.add()


model.compile(optimizer=SGD(lr=0.1), loss='mse')
model.fit(x, y, nb_epoch=5, batch_size=1)

model.summary()


model.evaluate(x, y, verbose=0)
model.predict(trn_data, batch_size=batch_size)         # return the probability for each class
model.get_weights()



##################### Useful functions #######################
def onehot(x): return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())

import bcolz
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]
save_array(model_path+ 'train_data.bc', trn_data)
save_array(model_path + 'valid_data.bc', val_data)
trn_data = load_array(model_path+'train_data.bc')
val_data = load_array(model_path+'valid_data.bc')