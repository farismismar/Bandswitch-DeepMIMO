#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:04:06 2019

@author: farismismar
"""

import os
import numpy as np
import h5py
import pandas as pd
import itertools

# This implements DeepMIMO_dataset{0,0}.user{0}.channel
os.chdir('/Users/farismismar/Desktop/DeepMIMO')

f = h5py.File('./DeepMIMO_Dataset_Generation v1.1/DeepMIMO Dataset/28_GHz_no_block_DeepMIMO_dataset.mat', 'r')

# Best practice
# Finds out the keys in the dataset.
#for key in f.keys():
#   print(key)
   
dataset = f['DeepMIMO_dataset']

# 1) Obtain {0,0} from Matlab
users = f[dataset[0,0]]['user']

# Best practice
# users[...] shows this is an array of 54481 references 
# - To dereference an h5py element, do f[.] around that element.
# Reference: https://groups.google.com/forum/#!topic/h5py/Q6wdqo3GnQ0

df = pd.DataFrame()
for userid in np.arange(54481).astype(int):
    
    # 2) Now the user in element i (or basically time step i)
    user_i = f[users[userid,0]]
    user_id_i = [userid]
    channel_i = np.transpose(user_i['channel'])
    loc_i = np.transpose(user_i['loc'][:])

    # 3) Take the first symbol as RS
    channel_i_rs = channel_i[:,0]        
    H_i = pd.DataFrame(channel_i_rs)
    H_i = [H_i.iloc[:,0].values, H_i.iloc[:,1].values]
    H_i = np.array(list(itertools.chain.from_iterable(H_i))) # all reals in H first then all imag in H
    
    # 4) Create an entry
    row_i = [user_id_i, H_i, loc_i.ravel()] # user ID | vec([real(H)]_i) | vec([imag(H)]_i) | x | y | z
    row_i = np.array(list(itertools.chain.from_iterable(row_i)))
    df_i = pd.DataFrame(row_i)
    df_i = df_i.T
        
    df = pd.concat([df, df_i])
    
# 5) Set the proper types
df.iloc[:,0] = df.iloc[:,0].astype(int) # id
df.iloc[:,-3:] = df.iloc[:,-3:].astype(float) # (x,y,z)

# 6) Write back to csv
df.to_csv('dataset/dataset_28_GHz.csv', index=False)