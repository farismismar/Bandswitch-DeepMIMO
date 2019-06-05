#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:08:03 2019

@author: farismismar
"""

import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.callbacks import History 
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";   # My NVIDIA GTX 1080 Ti FE GPU

os.chdir('/Users/farismismar/Desktop/DeepMIMO')

# 0) Some parameters
seed = 0
K_fold = 3
learning_rate = 0.01
max_users = 100

PTX_35 = 1 # in Watts for 3.5 GHz
PTX_28 = 1 # in Watts for 28 GHz

# 1) Read the data
# Add a few lines to caputre the seed for reproducibility.
random.seed(seed)
np.random.seed(seed)

df35 = pd.read_csv('dataset/dataset_3.5_GHz.csv')
df28 = pd.read_csv('dataset/dataset_28_GHz.csv')

# Truncate to the first max_users rows, for efficiency for now
df35 = df35.iloc[:max_users,:]
df28 = df28.iloc[:max_users,:]

# Map: 0 is ID; 1-256 are H real; 257-512 are Himag; 513-515 are x,y,z 

# 2) Perform data wrangling and construct the proper channel matrix H
H35_real = df35.iloc[:,1:256+1]
H35_imag = df35.iloc[:,257:512+1]
H35_loc = df35.iloc[:,513:]

H28_real = df28.iloc[:,1:256+1]
H28_imag = df28.iloc[:,257:512+1]
H28_loc = df28.iloc[:,513:]

# Before moving forward, check if the loc is equal
assert(np.all(df35.iloc[:,513:516] == df28.iloc[:,513:516]))

# Reset the column names of the imaginary H
H35_imag.columns = H35_real.columns
H28_imag.columns = H28_real.columns

H35 = H35_real + 1j * H35_imag
H28 = H28_real + 1j * H28_imag

del H35_real, H35_imag, H35_loc, H28_real, H28_imag, H28_loc

RSRP_35 = []
RSRP_28 = []

# Reconstruct H as 32 times 8
for i in np.arange(max_users):
    H35_i = np.array(H35.iloc[i,:]).reshape(32,8)
    H28_i = np.array(H28.iloc[i,:]).reshape(32,8)
    RSRP_35.append(PTX_35 * np.trace(np.matmul(H35_i, H35_i.conjugate().T)))
    RSRP_28.append(PTX_28 * np.trace(np.matmul(H28_i, H28_i.conjugate().T)))

# 3) Feature engineering: introduce RSRP mmWave and sub-6 and y

df35.loc[:,'RSRP_35'] = 10 * np.log10(RSRP_35).astype(float)
df28.loc[:,'RSRP_28'] = 10 * np.log10(RSRP_28).astype(float)

df = pd.concat([df35, df28], axis=1)

# TODO what is the HO criterion?
df['y'] = 

mX, nX = df.shape

# Perform a split 30-70
train, test = train_test_split(dataset, test_size=0.30, random_state=seed)
X_train = train.drop('y', axis=1)
X_test = test.drop('y', axis=1)
y_train = train.loc[:,'y']
y_test = train.loc[:,'y']

# Scale data
ss = StandardScaler()
X_train_sc = ss.fit_transform(X_train)
X_test_sc = ss.transform(X_test)

# Generate CNN    
model = Sequential()
model.add(Dense(32, input_dim=nX, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu')) # classifier, one output
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=learning_rate))

# The hyperparameters
activators = ['relu', 'sigmoid', 'softmax']
n_hiddens = [1,3]
hyperparameters = dict(activation = activators)

gs_clf = GridSearchCV(estimator=model, param_grid=hyperparameters, n_jobs=1, cv=K_fold)


with tf.device('/gpu:0'):
    grid_result = gs_clf.fit(X_train_sc, y_train)

# This is the best model
best_model_mlp = grid_result.best_params_
cnn_clf = grid_result.best_estimator_


with tf.device('/gpu:0'):
    y_pred = cnn_clf.predict(X_test_sc)

