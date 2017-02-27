# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 08:31:52 2017

@author: jiang_y
"""
##################### Data Preprecessing #######################
# Ref: https://keras.io/preprocessing/image/
from keras.preprocessing import image
gen=image.ImageDataGenerator()
gen.flow_from_directory(dirname, target_size=(224,224), class_mode='categorical', shuffle=True, batch_size=4)

val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
filenames = val_batches.filenames
val_classes = val_batches.classes

# change image dim ordering
from keras import backend as K
K.set_image_dim_ordering('th')

# VGG weights
# https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py
# https://keras.io/applications/#vgg16
from keras.applications.vgg16 import VGG16
model = VGG16()


##################### Models #######################
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D


model = Sequential()

model.add()
model.layers.pop()
model.summary()

model.compile(optimizer=SGD(lr=0.1), loss='mse')
model.fit(x, y, nb_epoch=5, batch_size=1)


model.evaluate(x, y, verbose=0)
model.predict(trn_data, batch_size=batch_size)         # return the probability for each class
preds = model.predict_classes(val_data, batch_size=batch_size)
probs = model.predict_proba(val_data, batch_size=batch_size)[:,0]
model.get_weights()

model.save_weights(model_path+'finetune1.h5')
model.load_weights(model_path+'finetune1.h5')

for layer in model.layers: layer.trainable=False        # fix all the layers

##################### Useful functions #######################
def onehot(x): return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())

import bcolz
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]
save_array(model_path+ 'train_data.bc', trn_data)
save_array(model_path + 'valid_data.bc', val_data)
trn_data = load_array(model_path+'train_data.bc')
val_data = load_array(model_path+'valid_data.bc')

