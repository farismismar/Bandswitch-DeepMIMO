#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 06:59:21 2019

@author: farismismar
"""

import random
import os
import numpy as np
import pandas as pd
import math

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";   # My NVIDIA GTX 1080 Ti FE GPU

import itertools
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV, train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from keras.backend.tensorflow_backend import set_session

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.ticker import MultipleLocator, FuncFormatter

from mpl_toolkits.mplot3d import Axes3D

import matplotlib2tikz
    
os.chdir('/Users/farismismar/Desktop/DeepMIMO')

scaler = StandardScaler()

# 0) Some parameters
seed = 0
K_fold = 2
learning_rate = 0.05
max_users = 54481
r_exploitation = 0.8
p_blockage = 0.4

p_randomness = 1 # 0 = all users start in 3.5

# in Mbps
rate_threshold_sub6 = 2.54 # [ 0.4300 0.8500 1.2700 1.7000 2.1200 2.5400]. 
rate_threshold_mmWave= 1.51 # 0.75,1.51,2.26,3.01,3.77,4.52

request_handover_threshold = (1 - p_randomness) * rate_threshold_sub6 + p_randomness * rate_threshold_mmWave  # this is y bar

# in ms
gap_fraction = 0.6 # rho

# in Watts
PTX_35 = 1 # in Watts for 3.5 GHz
PTX_28 = 1 # in Watts for 28 GHz

# speed:
v_s = 50 # km/h not pedestrian, but vehicular speeds.

delta_f_35 = 180e3 # Hz/PRB
delta_f_28 = 180e3 # Hz/PRB
N_SC_35 = 1
N_SC_28 = 1

mmWave_BW_multiplier = 3 # x sub-6
B_35 = N_SC_35 * delta_f_35
B_28 = N_SC_28 * delta_f_28 * mmWave_BW_multiplier
Nf = 7 # dB noise fig.

k_B = 1.38e-23 # Boltzmann
T = 290 # Kelvins

N_exploit = int(r_exploitation * max_users)

# 1) Read the data
# Add a few lines to caputre the seed for reproducibility.
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

def create_dataset():
    # Takes the three.csv files and merges them in a way that is useful for the Deep Learning.
    # regenerate the dataset for 3.5 (y,z = 8x4) and 28 (y, z = 64x4)
    df35 = pd.read_csv('dataset/dataset_3.5_GHz.csv')
    df28_b = pd.read_csv('dataset/dataset_28_GHz_blockage.csv')
    df28_nb = pd.read_csv('dataset/dataset_28_GHz.csv')
    
    # Truncate to the first max_users rows, for efficiency for now
    df35 = df35.iloc[:max_users,:]
    df28_b = df28_b.iloc[:max_users,:]
    df28_nb = df28_nb.iloc[:max_users,:]
    
    sub6_Y, sub6_Z = 8, 4
    mmWave_Y, mmWave_Z = 64, 4
    
    # Check that distances are similar
    assert(np.all(df28_b.iloc[:,-3:] == df28_nb.iloc[:,-3:]))
    
    # Based on blocking probability, create df28.
    p_b = np.random.binomial(1, p=p_blockage, size=max_users)
    df28 = df28_b.copy()
    df28.loc[(p_b==1),:] = df28_b.loc[(p_b == 1),:]
    df28.loc[(p_b==0),:] = df28_nb.loc[(p_b == 0),:]
    
    # Map: 0 is ID; 1-YZ+1 are H real; YZ+1-2YZ+1 are Himag; last three are x,y,z 
    # 2) Perform data wrangling and construct the proper channel matrix H
    H35_real = df35.iloc[:,1:(sub6_Y*sub6_Z+1)]
    H35_imag = df35.iloc[:,(sub6_Y*sub6_Z+1):(2*sub6_Y*sub6_Z+1)]
    H35_loc = df35.iloc[:,-3:]
        
    H28_real = df28.iloc[:,1:(mmWave_Y*mmWave_Z+1)]
    H28_imag = df28.iloc[:,(mmWave_Y*sub6_Z+1):(2*mmWave_Y*mmWave_Z+1)]
    H28_loc = df28.iloc[:,-3:]    
       
    # Before moving forward, check if the loc at time t is equal
    df35 = df35.rename(columns={df35.columns[-3]:  'lon', 
                         df35.columns[-2]:  'lat', 
                         df35.columns[-1]:  'height'})

    df28 = df28.rename(columns={df28.columns[-3]:  'lon', 
                         df28.columns[-2]:  'lat', 
                         df28.columns[-1]:  'height'})
    
    assert(np.all(df35.iloc[:,-3:] == df28.iloc[:,-3:]))
    
    # Reset the column names of the imaginary H
    H35_imag.columns = H35_real.columns
    H28_imag.columns = H28_real.columns
    
    H35 = H35_real + 1j * H35_imag
    H28 = H28_real + 1j * H28_imag
    
    del H35_loc, H28_real, H28_imag, H28_loc
    
    F_35 = compute_bf_codebook(My=sub6_Y, Mz=sub6_Z, f_c=3.5e9)
    F_28 = compute_bf_codebook(My=mmWave_Y, Mz=mmWave_Z, f_c=28e9)
    
    channel_gain_35 = []
    channel_gain_28 = []
    
    # Compute the channel gain |h*f|
    # Beamforming is now both vertical and horizontal
    for i in np.arange(max_users):
        h35_i = np.array(H35.iloc[i,:])
        h28_i = np.array(H28.iloc[i,:])
        channel_gain_35.append(compute_optimal_gain_bf_vector(h35_i, F_35))
        channel_gain_28.append(compute_optimal_gain_bf_vector(h28_i, F_28))
    
    # 3) Feature engineering: introduce RSRP mmWave and sub-6 and y
    channel_gain_28 = np.array(channel_gain_28).astype(float)
    channel_gain_35 = np.array(channel_gain_35).astype(float)
    
    # Get rid of unwanted columns in 3.5
    df35 = df35[['0', 'lon', 'lat', 'height']]
    df35.columns = ['user_id', 'lon', 'lat', 'height']

    df = df35.copy()    
    df.loc[:,'P_RX_35'] = 10*np.log10(PTX_35 * 1e3 * channel_gain_35)
    df.loc[:,'P_RX_28'] = 10*np.log10(PTX_28 * 1e3 * channel_gain_28)
    
    df = df.iloc[:max_users,:]
    df = df[['user_id', 'lon', 'lat', 'height', 'P_RX_35', 'P_RX_28']]
    df.to_csv('dataset.csv', index=False)
    
    return df

def compute_optimal_gain_bf_vector(h, F):
    M, MK = F.shape

    max_gain = 0

    for code_index in np.arange(MK):
        f_i = F[:,code_index]
        channel_gain = abs(np.vdot(h, f_i)) ** 2
        if (channel_gain > max_gain):
            max_gain = channel_gain
            
    return channel_gain
    
def compute_bf_codebook(My, Mz, f_c, k_oversampling=1):
    Fy = np.zeros([My, My*k_oversampling], dtype=complex) # F is M rows by Mk columns, where M corresponds to the antennas in the horizontal direction

    theta_y_n = math.pi * np.arange(start=0., stop=1., step=1./(k_oversampling*My))

    for n in np.arange(My*k_oversampling):
        f_n = _compute_bf_vector(f_c, theta_y_n[n], My)
        Fy[:,n] = f_n
            
    Fz = np.zeros([Mz, Mz*k_oversampling], dtype=complex) # F is M rows by Mk columns, where M corresponds to the antennas in the horizontal direction

    theta_z_n = math.pi * np.arange(start=0., stop=1., step=1./(k_oversampling*Mz))

    for n in np.arange(Mz*k_oversampling):
        f_n = _compute_bf_vector(f_c, theta_z_n[n], Mz)
        Fz[:,n] = f_n

    F = np.kron(Fz, Fy)
    
    return F

def _compute_bf_vector(f_c, theta, M_ULA):
    # Create DFT beamforming codebook
    c = 299792458 # speed of light
    wavelength = c / f_c
    
    d = wavelength / 2. # antenna spacing 
    k = 2. * math.pi / wavelength

    exponent = 1j * k * d * math.cos(theta) * np.arange(M_ULA)
    
    f = 1. / math.sqrt(M_ULA) * np.exp(exponent)
    
    return f

def get_misclassification_error(y_test, y_pred, y_score):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp  = cm.ravel()
    
    mu = (fp + fn) / (fp + fn + tn + tp)
    
    return cm, mu

def plot_confusion_matrix(y_test, y_pred, y_score):
    # Compute confusion matrix
    classes = [0,1]
    class_names = ['Deny','Grant']
    normalize = False
    
    cm, _  = get_misclassification_error(y_test, y_pred, y_score)
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    fig = plt.figure(figsize=(10.24, 7.68))
    ax = fig.gca()
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 30
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    ax.matshow(cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
    plt.xticks = np.arange(cm.shape[1])
    plt.yticks = np.arange(cm.shape[0])

    # label the ticks with the respective list entries
    ax.set_xticklabels(['']+class_names)
    ax.set_yticklabels(['']+class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2.
    for i, j in itertools.product(np.arange(cm.shape[0]), np.arange(cm.shape[1])):
        ax.text(x=j, y=i, s=format(cm[i, j], fmt),
                 horizontalalignment="center", va='center',
                 color="white" if cm[i, j] > thresh else "black")
 
    plt.xlabel(r'\textbf{Predicted label}')
    plt.ylabel(r'\textbf{True label}')
    
    plt.tight_layout()
    plt.savefig('figures/conf_matrix_{}.pdf'.format(p_randomness), format='pdf')

def plot_joint_pdf(X, Y):
    fig = plt.figure(figsize=(10.24, 7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 30
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']   
    
    num_bins = 50
    pdf, X_bin_edges, Y_bin_edges = np.histogram2d(X, Y, bins=(num_bins, num_bins), normed=True)
    pdf = pdf.T 

    ax = plt.gca(projection="3d")
    x, y = np.meshgrid(X_bin_edges, Y_bin_edges)

    ax.plot_surface(x[:num_bins, :num_bins], y[:num_bins, :num_bins], pdf[:num_bins, :num_bins], cmap='Spectral_r')

    ax.set_xlabel('{} [Mbps]'.format(X.name))
    ax.set_ylabel('{} [Mbps]'.format(Y.name))
    ax.set_zlabel('Joint Throughput pdf')
    ax.xaxis.labelpad=30
    ax.yaxis.labelpad=30
    ax.zaxis.labelpad=30
    plt.tight_layout()
    plt.savefig('figures/joint_throughput_pdf_{}.pdf'.format(p_randomness), format='pdf')
    matplotlib2tikz.save('figures/joint_throughput_pdf_{}.tikz'.format(p_randomness))


def plot_pdf(data1, label1, data2, label2):
    fig = plt.figure(figsize=(10.24, 7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 40
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']   
    
    labels = [label1, label2]

    num_bins = 50
    counts, bin_edges = np.histogram(data1, bins=num_bins, density=True)
    pdf = counts #np.cumsum(counts) / counts.sum()

    lw = 2    
    plt.xlabel('Coherence time (ms)')
    plt.grid(True, axis='both', which='both')
    ax = fig.gca()
    plot1, = ax.plot(bin_edges[1:], pdf, linewidth=lw)
    ax.set_ylabel('sub-6 Coherence time pdf')
    
    counts, bin_edges = np.histogram(data2, bins=num_bins, density=True)
    pdf = counts #np.cumsum(counts) / counts.sum()
    ax_sec = ax.twinx()
    plot2, = ax_sec.plot(bin_edges[1:], pdf, color='red', linewidth=lw)
    
    plt.legend([plot1, plot2], labels, loc="best")
    ax_sec.set_ylabel('mmWave Coherence time pdf')
    plt.tight_layout()
    plt.savefig('figures/coherence_time_{}.pdf'.format(p_randomness), format='pdf')
    matplotlib2tikz.save('figures/coherence_time_{}.tikz'.format(p_randomness))
    
def plot_throughput_cdf(T):
    fig = plt.figure(figsize=(10.24, 7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 40
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']   
    
    labels = T.columns

    num_bins = 50
    i = 0
    for data in T:
        data_ = T[data]

        counts, bin_edges = np.histogram(data_, bins=num_bins, density=True)
        cdf = np.cumsum(counts) / counts.sum()
        lw = 1 + 0.2*i
        i += 1
        ax = fig.gca()
        if data == 'Optimal':
            style = '--'
        elif data == 'Proposed':
#            lw = 3.5
            style = '+-'
        else:
            style = '-'
        ax.plot(bin_edges[1:], cdf, style, linewidth=lw)
    
    plt.legend(labels, loc="best")    
    plt.grid()
    plt.xlabel('Throughput (Mbps)')
    plt.ylabel('Throughput CDF')
    plt.tight_layout()
    plt.savefig('figures/throughputs_{}.pdf'.format(p_randomness), format='pdf')
    matplotlib2tikz.save('figures/throughputs_{}.tikz'.format(p_randomness))

def plot_throughput_pdf(T):
    fig = plt.figure(figsize=(10.24, 7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 40
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']   
    
    labels = [] 

    num_bins = 40
    for data in T:
        data_ = T[data]

        counts, bin_edges = np.histogram(data_, bins=num_bins, density=True)
        pdf = counts #np.cumsum(counts) / counts.sum()
        ax = fig.gca()
        if data == 'mmWave only':
            style = 'r-'
            labels.append(data)
        elif data == 'Sub-6 only':
            style = 'b-'
            labels.append(data)
        else:
            continue
        ax.plot(bin_edges[1:], pdf, style, linewidth=2)
    
    plt.legend(labels, loc="best")    
    plt.grid()
    plt.xlabel('Throughput (Mbps)')
    plt.ylabel('Throughput pdf')
    plt.tight_layout()
    plt.savefig('figures/throughputs_pdf_{}.pdf'.format(p_randomness), format='pdf')
    matplotlib2tikz.save('figures/throughputs_pdf_{}.tikz'.format(p_randomness))
    
def plot_primary(X,Y, title, xlabel, ylabel, filename='plot.pdf'):
    fig = plt.figure(figsize=(10.24,7.68))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 40
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    plt.xlabel(xlabel)
    
    ax = fig.gca()
    ax.set_autoscaley_on(True)
    
    plot_, = ax.plot(X, Y, 'k^-') #, label='ROC')

    ax.set_ylabel(ylabel)
    
    plt.grid(True)
    fig.tight_layout()
    plt.savefig('figures/plot_{0}{1}.pdf'.format(p_randomness, filename), format='pdf')
    matplotlib2tikz.save('figures/plot_{0}{1}.tikz'.format(p_randomness, filename))
    
#    plt.show()

##############################################################################
def create_mlp(input_dimension, hidden_dimension, n_hidden):
    n_classes = 1
    
    model = Sequential()
    model.add(Dense(units=hidden_dimension, use_bias=True, input_dim=input_dimension, activation='sigmoid'))
    for h in np.arange(n_hidden):
        model.add(Dense(units=hidden_dimension, use_bias=True, activation='sigmoid'))
    model.add(Dense(units=n_classes, input_dim=hidden_dimension, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer = Adam(lr=learning_rate), metrics=['accuracy'])

    return model

def train_classifier(df, r_training=0.8):
    dataset = df.copy()
    
    training, test = train_test_split(dataset, train_size=r_training, random_state=seed)
    
    X_train = training.drop('y', axis=1)
    y_train = training['y']
    X_test = test.drop('y', axis=1)
    y_test = test['y']
    
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    mX, nX = X_train.shape
    
    model = KerasClassifier(build_fn=create_mlp, verbose=0, epochs=5, batch_size=32)

    # The hyperparameters
    hidden_dims = [3,5,10]
    n_hiddens = [1,3,5] #[1,2,3]
    K_fold = 2
    hyperparameters = dict(input_dimension=[nX], hidden_dimension=hidden_dims, n_hidden=n_hiddens)
    grid = GridSearchCV(estimator=model, param_grid=hyperparameters, n_jobs=1, cv=K_fold)
    
    gpu_available = tf.test.is_gpu_available()
    if (gpu_available == False):
        print('WARNING: No GPU available.  Will continue with CPU.')
        
    with tf.device('/gpu:0'):
        grid_result = grid.fit(X_train_sc, y_train, class_weight=class_weights)
    
    # This is the best model
    print(grid_result.best_params_)
    clf = grid_result.best_estimator_

    # clf.model.get_config()
#    grid_result.best_estimator_.model.save("model_fc.h5") 
        
    with tf.device('/gpu:0'):
        y_pred = clf.predict(X_test_sc)
        y_score = clf.predict_proba(X_test_sc)
    try:
        roc_auc = roc_auc_score(y_test, y_score[:,1])
        print('The Training ROC AUC for this classifier is {:.6f}'.format(roc_auc))
    except:
        print('The Training ROC AUC for this classifier is N/A')

    return [y_pred, y_score, clf]

def predict_handover(df, clf, r_training):
    # The exploit phase data
    y_test = df['y']
    X_test = df.drop(['y'], axis=1)
    
    X_test_sc = scaler.transform(X_test)
    
    with tf.device('/gpu:0'):
        y_pred = clf.predict(X_test_sc)
        y_score = clf.predict_proba(X_test_sc)
    
    try:
        # Compute area under ROC curve
        roc_auc = roc_auc_score(y_test, y_score[:,1])
        print('The ROC AUC for the exploitation period is {:.6f}'.format(roc_auc))
    
        # Save the value
        f = open("figures/output_fc_{}.txt".format(p_randomness), 'a')
        f.write('r_exploitation {0}, r_training {1}, ROC {2:.6f}\n'.format(r_exploitation, r_training, roc_auc))
        f.close()

        y_pred=pd.DataFrame(y_pred)
      
    except:
       print('The ROC AUC for the exploitation period is N/A')
       y_pred = None
       roc_auc = None
       
    return y_pred, roc_auc
##############################################################################
    
def get_beam_training_time(df, freq=28e9, horiz_beams=32, vertical_beams=8):
    return 10e-3 * horiz_beams * vertical_beams # 10 us in ms per beam.

def get_coherence_time(df, My, freq):
    # Returns beam coherence time in ms.
    c = 299792458 # speed of light
    
    BS_x, BS_y, BS_z = [235.504198, 489.503816, 6]
    np.random.seed(seed)

    n = df.shape[0]    
    
    # Obtain D
    # alpha AoA equivalent random(0, pi) or 30 to 150 degrees

    D = ((df['lon'] - BS_x) ** 2 + (df['lat'] - BS_y) ** 2 + (df['height'] - BS_z) ** 2) ** 0.5
    Theta_n = 102 / My * math.pi/180 # beamwidth approximation for ULA ### 64 antennas in the aziumuth direction # 3 dB BW of antenna
    alpha = np.random.uniform(0, math.pi, size=n)
    T_B = D / (v_s * 1000/3600 * np.sin(alpha)) * Theta_n / 2.

    T_beam = np.array(T_B) * 1e3 # in ms
    T_beam = np.percentile(T_beam, 1) # take the 1st percentile of coherence
    
    if freq >= 28e9:
        print('INFO: mmWave mean channel coherence time is {} ms'.format(T_beam.mean()))
        return T_beam        
  
    T_ofdm = np.ones(n) * c / (freq * v_s * 1000/3600) * 1e3 # in ms

    T = np.minimum(T_ofdm, T_beam)

    print('INFO: sub-6 mean channel coherence time is {} ms'.format(T.mean()))
    return T

#df_ = create_dataset() # only uncomment for the first run, when the channel consideration changed.
df_ = pd.read_csv('dataset.csv')

df = df_.iloc[:max_users,:]
del df_

# Feature engineering: add SNR to the computation:
noise_floor_35 = k_B * T * delta_f_35 * 1e3
noise_floor_28 = k_B * T * delta_f_28 * mmWave_BW_multiplier * 1e3 # in mW

noise_power_35 = 10 ** (Nf/10.) * noise_floor_35
noise_power_28 = 10 ** (Nf/10.) * noise_floor_28 

# Instantaneous rates (Shannon)
df['Capacity_35'] = B_35*np.log2(1 + 10**(df['P_RX_35']/10.) / noise_power_35) / 1e6
df['Capacity_28'] = B_28*np.log2(1 + 10**(df['P_RX_28']/10.) / noise_power_28) / 1e6

# 3D Plot PDF for Capacity_35 and Capacity_28
plot_joint_pdf(df['Capacity_35'], df['Capacity_28'])

df = df[['lon', 'lat', 'height', 'Capacity_35', 'Capacity_28']]

user_mask = np.random.binomial(1, p_randomness, size=max_users) # 0 == user is 3.5, 1 == user is mmWave.

# Source and Target are instantaneous rates.
df.loc[user_mask==0, 'Source'] = df.loc[user_mask==0, 'Capacity_35']
df.loc[user_mask==1, 'Source'] = df.loc[user_mask==1, 'Capacity_28']
df.loc[user_mask==0, 'Target'] = df.loc[user_mask==0, 'Capacity_28']
df.loc[user_mask==1, 'Target'] = df.loc[user_mask==1, 'Capacity_35']

# Compute the Effective Achievable Rates
coherence_time_sub6 = get_coherence_time(df, My=8, freq=3.5e9)
coherence_time_mmWave = get_coherence_time(df, My=64, freq=28e9) 

#plot_pdf(coherence_time_mmWave, 'mmWave', coherence_time_sub6, 'sub-6')
coherence_time_mmWave = np.percentile(coherence_time_mmWave, 1)
coherence_time_sub6 = np.mean(coherence_time_sub6)

gap_duration_sub6 = gap_fraction * coherence_time_sub6
gap_duration_mmWave  = gap_fraction * coherence_time_mmWave

beam_training_penalty_mmWave = get_beam_training_time(df, freq=28e9, horiz_beams=8, vertical_beams=32)
beam_training_penalty_sub6 = get_beam_training_time(df, freq=2.1e9, horiz_beams=8, vertical_beams=8)

# Write the formulas in Paper
coeff_sub6_no_ho = (coherence_time_sub6 - beam_training_penalty_sub6) / coherence_time_sub6
coeff_mmWave_no_ho = (coherence_time_mmWave - beam_training_penalty_mmWave) / coherence_time_mmWave
coeff_sub6_ho = (coherence_time_sub6 - beam_training_penalty_sub6 - gap_duration_sub6) / coherence_time_sub6
coeff_mmWave_ho = (coherence_time_mmWave - beam_training_penalty_mmWave - gap_duration_mmWave) / coherence_time_mmWave

df.to_csv('figures/dataset_rates_{}.csv'.format(p_randomness))

##############################################################################
df['Source_is_3.5'] = (df['Source'] == df['Capacity_35']) + 0
df['Source_is_28'] = (df['Source'] == df['Capacity_28']) + 0

exploit_indices = np.random.choice(df.shape[0], N_exploit, replace=False)

sub_6_capacities = df.loc[exploit_indices, 'Capacity_35'].copy()
mmWave_capacities = df.loc[exploit_indices, 'Capacity_28'].copy()

# Change the order of columns to put 
column_order = ['lon', 'lat', 'height', 'Source', 'Target', 'Source_is_3.5', 'Source_is_28']
df = df[column_order]

##############################################################################
# 1) Optimal algorithm
##############################################################################
df_optimal = df.copy()
df_optimal_ = df.copy()

# Now, apply the handover algorithm
# and compute the Effective Achievable Rate but no penalty for handover

a = df_optimal_.loc[(df_optimal_['Source_is_3.5'] == 1), 'Source'] * coeff_sub6_no_ho
b = df_optimal_.loc[(df_optimal_['Source_is_3.5'] == 1), 'Target'] * coeff_mmWave_no_ho
c = df_optimal_.loc[(df_optimal_['Source_is_28'] == 1), 'Source'] * coeff_mmWave_no_ho
d = df_optimal_.loc[(df_optimal_['Source_is_28'] == 1), 'Target'] * coeff_sub6_no_ho

# The NaNs here are due to p_randomness values.
df_optimal_ = pd.DataFrame([a, b, c, d]).T
df_optimal_.fillna(0, axis=1, inplace=True)

# Choose the max rate regardless
df_optimal.loc[:,'Capacity_Optimal'] = df_optimal_.apply(np.max, axis=1)
      
# Sample r_exploit data randomly from df_optimal
benchmark_data_optimal = df_optimal.iloc[exploit_indices, :]

del df_optimal, a, b, d, df_optimal_

##############################################################################
# 2) Legacy algorithm
##############################################################################
df_legacy = df.copy()

# Handover is based on raw Shannon rates.
df_legacy.loc[:, 'HO_requested'] = (df_legacy.loc[:, 'Source'] < request_handover_threshold) + 0
df_legacy.loc[:, 'y'] = (df_legacy.loc[:,'Target'] >= df_legacy.loc[:,'Source']) + 0

# No handover request means no handover granted
df_legacy.loc[df_legacy['HO_requested'] == 0, 'y'] = 0

# Now, apply the handover algorithm
# and compute the Effective Achievable Rate

# Based on y_bar, if there was no handover, put the source effective rates back
df_legacy.loc[(df_legacy['HO_requested'] == 0) & (df_legacy['Source_is_3.5'] == 1), 'Capacity_Legacy'] = df_legacy.loc[(df_legacy['HO_requested'] == 0) & (df_legacy['Source_is_3.5'] == 1), 'Source'] * coeff_sub6_no_ho # no handover requested.
df_legacy.loc[(df_legacy['HO_requested'] == 0) & (df_legacy['Source_is_28'] == 1), 'Capacity_Legacy'] = df_legacy.loc[(df_legacy['HO_requested'] == 0) & (df_legacy['Source_is_28'] == 1), 'Source'] * coeff_mmWave_no_ho # no handover requested.

# Handover requested, but denied.  Therefore, the source rate penalized by the gap
df_legacy.loc[(df_legacy['HO_requested'] == 1) & (df_legacy['y'] == 0) & (df_legacy['Source_is_3.5'] == 1), 'Capacity_Legacy'] = df_legacy.loc[(df_legacy['HO_requested'] == 1) & (df_legacy['y'] == 0) & (df_legacy['Source_is_3.5'] == 1), 'Source'] * coeff_sub6_ho # handover requested but denied, the throughput is the source.
df_legacy.loc[(df_legacy['HO_requested'] == 1) & (df_legacy['y'] == 0) & (df_legacy['Source_is_28'] == 1), 'Capacity_Legacy'] = df_legacy.loc[(df_legacy['HO_requested'] == 1) & (df_legacy['y'] == 0) & (df_legacy['Source_is_28'] == 1), 'Source'] * coeff_mmWave_ho # handover requested but denied, the throughput is the source.

# Handover requested, and granted.  Therefore, the target rate penalized by the gap
df_legacy.loc[(df_legacy['HO_requested'] == 1) & (df_legacy['y'] == 1) & (df_legacy['Source_is_3.5'] == 1), 'Capacity_Legacy'] = df_legacy.loc[(df_legacy['HO_requested'] == 1) & (df_legacy['y'] == 1) & (df_legacy['Source_is_3.5'] == 1), 'Target'] * coeff_sub6_ho # handover requested and granted, the throughput is the target.
df_legacy.loc[(df_legacy['HO_requested'] == 1) & (df_legacy['y'] == 1) & (df_legacy['Source_is_28'] == 1), 'Capacity_Legacy'] = df_legacy.loc[(df_legacy['HO_requested'] == 1) & (df_legacy['y'] == 1) & (df_legacy['Source_is_28'] == 1), 'Target'] * coeff_mmWave_ho # handover requested and granted, the throughput is the target.
##

# Sample r_exploit data randomly from df_legacy
benchmark_data_legacy = df_legacy.iloc[exploit_indices, :]

del df_legacy

##############################################################################
# 3) Blind handover algorithm
##############################################################################
df_blind = df.copy()

df_blind['HO_requested'] = pd.DataFrame((df_blind.loc[:,'Source'] <= request_handover_threshold), dtype=int)
df_blind['y'] = 1

# No handover request means no handover granted
df_blind.loc[df_blind['HO_requested'] == 0, 'y'] = 0

# Now, apply the handover algorithm
# and compute the Effective Achievable Rate
#df_blind.loc[(df_blind['y'] == 0) & (df_blind['Source_is_3.5'] == 1), 'Capacity_Blind'] = df_blind.loc[(df_blind['y'] == 0)  & (df_blind['Source_is_3.5'] == 1), 'Source'] * coeff_sub6_no_ho # no handover, the throughput is the source.
#df_blind.loc[(df_blind['y'] == 0) & (df_blind['Source_is_28'] == 1), 'Capacity_Blind'] = df_blind.loc[(df_blind['y'] == 0)  & (df_blind['Source_is_28'] == 1), 'Source'] * coeff_mmWave_no_ho # no handover, the throughput is the source.

# Based on y_bar, if there was no handover, put the source effective rates back
df_blind.loc[(df_blind['HO_requested'] == 0) & (df_blind['Source_is_3.5'] == 1), 'Capacity_Blind'] = df_blind.loc[(df_blind['HO_requested'] == 0) & (df_blind['Source_is_3.5'] == 1), 'Source'] * coeff_sub6_no_ho # no handover, the throughput is the source but no gap.
df_blind.loc[(df_blind['HO_requested'] == 0) & (df_blind['Source_is_28'] == 1), 'Capacity_Blind'] = df_blind.loc[(df_blind['HO_requested'] == 0) & (df_blind['Source_is_28'] == 1), 'Source'] * coeff_mmWave_no_ho # no handover, the throughput is the source but no gap.

# Handover requested, but denied.  Therefore, the source rate penalized but no gap
df_blind.loc[(df_blind['HO_requested'] == 1) & (df_blind['y'] == 0) & (df_blind['Source_is_3.5'] == 1), 'Capacity_Blind'] = df_blind.loc[(df_blind['HO_requested'] == 1) & (df_blind['y'] == 0) & (df_blind['Source_is_3.5'] == 1), 'Source'] * coeff_sub6_no_ho # no handover, the throughput is the source but no gap.
df_blind.loc[(df_blind['HO_requested'] == 1) & (df_blind['y'] == 0) & (df_blind['Source_is_28'] == 1), 'Capacity_Blind'] = df_blind.loc[(df_blind['HO_requested'] == 1) & (df_blind['y'] == 0) & (df_blind['Source_is_28'] == 1), 'Source'] * coeff_mmWave_no_ho # no handover, the throughput is the source but no gap.

# Handover requested, and granted.  Therefore, the target rate penalized but no gap
df_blind.loc[(df_blind['HO_requested'] == 1) & (df_blind['y'] == 1) & (df_blind['Source_is_3.5'] == 1), 'Capacity_Blind'] = df_blind.loc[(df_blind['HO_requested'] == 1) & (df_blind['y'] == 1) & (df_blind['Source_is_3.5'] == 1), 'Target'] * coeff_mmWave_no_ho # blind handover, the throughput is the target but no gap.
df_blind.loc[(df_blind['HO_requested'] == 1) & (df_blind['y'] == 1) & (df_blind['Source_is_28'] == 1), 'Capacity_Blind'] = df_blind.loc[(df_blind['HO_requested'] == 1) & (df_blind['y'] == 1) & (df_blind['Source_is_28'] == 1), 'Target'] * coeff_sub6_no_ho # blind handover, the throughput is the target but no gap.

##

# Sample r_exploit data randomly from df_blind
benchmark_data_blind = df_blind.iloc[exploit_indices, :]

del df_blind

##############################################################################
# 4) Proposed algorithm
##############################################################################

# The height column must be deleted here before prediction is made
height = df['height']
df_proposed = df.drop(['height', 'Source_is_28'], axis=1) # delete the 28 column since it is equal to not 3.5.

df_proposed['HO_requested'] = pd.DataFrame((df_proposed.loc[:,'Source'] <= request_handover_threshold), dtype=int)
df_proposed.loc[:, 'y'] = (df_proposed.loc[:,'Target'] >= df_proposed.loc[:,'Source']) + 0

# No handover request means no handover granted
df_proposed.loc[df_proposed['HO_requested'] == 0, 'y'] = 0

if (p_randomness == 0 or p_randomness == 1):
    df_proposed = df_proposed.drop(['Source_is_3.5'], axis=1) # these two values will make the column of a single value.

# Use this for the exploitation
train_valid, benchmark_data_proposed = train_test_split(df_proposed, test_size=r_exploitation, random_state=seed)

train_indices = pd.Int64Index(np.arange(df.shape[0])).difference(exploit_indices)
train_valid = df_proposed.iloc[train_indices, :]

benchmark_data_proposed = df_proposed.iloc[exploit_indices, :]

roc_graphs = pd.DataFrame()
misclass_graphs = pd.DataFrame()

roc_auc_values = []
misclass_error_values = []

min_r_training = 1
min_score = np.inf
best_clf = None
X = [1e-3, 3e-3,5e-3,7e-3, 1e-2,3e-2,5e-2,7e-2,1e-1,0.4] # np.arange(1,10,1)/10.
for r_t in X:
    try:
        [y_pred, y_score, clf] = train_classifier(train_valid, r_t)
        y_pred_proposed, score = predict_handover(benchmark_data_proposed, clf, r_t)
        y_score_proposed = clf.predict_proba(benchmark_data_proposed.drop(['y'], axis=1))
        y_test_proposed = benchmark_data_proposed['y']
        _, mu = get_misclassification_error(y_test_proposed, y_pred_proposed, y_score_proposed)
        print('The misclassification error in the exploitation period is {:.6f}%.'.format(mu*100))
#        fpr, tpr, score = generate_roc(y_test_proposed, y_score_proposed[:,1])
        if (mu < min_score):
            min_score = mu
            min_r_training = r_t
            best_clf = clf
            
        roc_auc_values.append(score)
        misclass_error_values.append(mu)
        
        roc_graphs = pd.concat([roc_graphs, pd.DataFrame(roc_auc_values)], axis=1)
        misclass_graphs = pd.concat([misclass_graphs, pd.DataFrame(misclass_error_values)], axis=1)
        
    except:
        roc_auc_values.append(np.nan)
        misclass_error_values.append(np.nan)
        pass

roc_graphs.to_csv('figures/roc_output_{}.csv'.format(p_randomness), index=False)
misclass_graphs.to_csv('figures/misclass_output_{}.csv'.format(p_randomness), index=False)

# Now generate data with the best classifier.
y_pred_proposed, _ = predict_handover(benchmark_data_proposed, best_clf, min_r_training)
y_score_proposed = best_clf.predict_proba(benchmark_data_proposed.drop(['y'], axis=1))
y_test_proposed = benchmark_data_proposed['y']

# Put back the height column
benchmark_data_proposed['height'] = height

# Put back the Source data
benchmark_data_proposed['Source_is_3.5'] = df.loc[benchmark_data_proposed.index, 'Source_is_3.5']
benchmark_data_proposed['Source_is_28'] = df.loc[benchmark_data_proposed.index, 'Source_is_28']

# Penalize the throughput rates aka Effective Achievable Rate
# Use the same formula as the blind formula

# Based on y_bar, if there was no handover, put the source effective rates back
benchmark_data_proposed.loc[(benchmark_data_proposed['HO_requested'] == 0) & (benchmark_data_proposed['Source_is_3.5'] == 1), 'Capacity_Proposed'] = benchmark_data_proposed.loc[(benchmark_data_proposed['HO_requested'] == 0) & (benchmark_data_proposed['Source_is_3.5'] == 1), 'Source'] * coeff_sub6_no_ho # no handover, the throughput is the source but no gap.
benchmark_data_proposed.loc[(benchmark_data_proposed['HO_requested'] == 0) & (benchmark_data_proposed['Source_is_28'] == 1), 'Capacity_Proposed'] = benchmark_data_proposed.loc[(benchmark_data_proposed['HO_requested'] == 0) & (benchmark_data_proposed['Source_is_28'] == 1), 'Source'] * coeff_mmWave_no_ho # no handover, the throughput is the source but no gap.

# Handover requested, but denied.  Therefore, the source rate penalized but no gap
benchmark_data_proposed.loc[(benchmark_data_proposed['HO_requested'] == 1) & (benchmark_data_proposed['y'] == 0) & (benchmark_data_proposed['Source_is_3.5'] == 1), 'Capacity_Proposed'] = benchmark_data_proposed.loc[(benchmark_data_proposed['HO_requested'] == 1) & (benchmark_data_proposed['y'] == 0) & (benchmark_data_proposed['Source_is_3.5'] == 1), 'Source'] * coeff_sub6_no_ho # no handover, the throughput is the source but no gap.
benchmark_data_proposed.loc[(benchmark_data_proposed['HO_requested'] == 1) & (benchmark_data_proposed['y'] == 0) & (benchmark_data_proposed['Source_is_28'] == 1), 'Capacity_Proposed'] = benchmark_data_proposed.loc[(benchmark_data_proposed['HO_requested'] == 1) & (benchmark_data_proposed['y'] == 0) & (benchmark_data_proposed['Source_is_28'] == 1), 'Source'] * coeff_mmWave_no_ho # no handover, the throughput is the source but no gap.

# Handover requested, and granted.  Therefore, the target rate penalized but no gap
benchmark_data_proposed.loc[(benchmark_data_proposed['HO_requested'] == 1) & (benchmark_data_proposed['y'] == 1) & (benchmark_data_proposed['Source_is_3.5'] == 1), 'Capacity_Proposed'] = benchmark_data_proposed.loc[(benchmark_data_proposed['HO_requested'] == 1) & (benchmark_data_proposed['y'] == 1) & (benchmark_data_proposed['Source_is_3.5'] == 1), 'Target'] * coeff_mmWave_no_ho # blind handover, the throughput is the target but no gap.
benchmark_data_proposed.loc[(benchmark_data_proposed['HO_requested'] == 1) & (benchmark_data_proposed['y'] == 1) & (benchmark_data_proposed['Source_is_28'] == 1), 'Capacity_Proposed'] = benchmark_data_proposed.loc[(benchmark_data_proposed['HO_requested'] == 1) & (benchmark_data_proposed['y'] == 1) & (benchmark_data_proposed['Source_is_28'] == 1), 'Target'] * coeff_sub6_no_ho # blind handover, the throughput is the target but no gap.


##

##############################################################################
# Plotting
##############################################################################

plot_primary(X, roc_auc_values, 'ROC vs Training', r'$r_\text{training}$', 'ROC AUC', filename='roc_vs_training_{}.pdf'.format(p_randomness))
plot_primary(X, 100*np.array(misclass_error_values), '$\mu vs Training', r'$r_\text{training}$', r'$\mu$ [\%]', filename='misclass_vs_training_{}.pdf'.format(p_randomness))
plot_confusion_matrix(y_test_proposed, y_pred_proposed, y_score_proposed)

# Put the coherence time penalty for no handover regardess
sub_6_capacities.iloc[:] *= coeff_sub6_no_ho
mmWave_capacities.iloc[:] *= coeff_mmWave_no_ho

benchmark_data_optimal = benchmark_data_optimal.reset_index().drop(['index'], axis=1)
benchmark_data_proposed = benchmark_data_proposed.reset_index().drop(['index'], axis=1)
benchmark_data_legacy  = benchmark_data_legacy.reset_index().drop(['index'], axis=1)
benchmark_data_blind = benchmark_data_blind.reset_index().drop(['index'], axis=1)
benchmark_data_proposed = benchmark_data_proposed.reset_index().drop(['index'], axis=1)
sub_6_capacities = sub_6_capacities.reset_index().drop(['index'], axis=1)
mmWave_capacities = mmWave_capacities.reset_index().drop(['index'], axis=1)

data = pd.concat([benchmark_data_optimal['Capacity_Optimal'], benchmark_data_proposed['Capacity_Proposed'], benchmark_data_proposed['HO_requested'], benchmark_data_legacy['Capacity_Legacy'], benchmark_data_blind['Capacity_Blind'], sub_6_capacities['Capacity_35'], mmWave_capacities['Capacity_28']], axis=1, ignore_index=True)
data.columns = ['Optimal', 'Proposed', 'HO_requested', 'Legacy', 'Blind', 'Sub-6 only', 'mmWave only']
data.to_csv('figures/dataset_post_{}.csv'.format(p_randomness), index=False)

plot_throughput_pdf(data)

data = data[['Optimal', 'Proposed', 'Legacy', 'Blind']]
data.dropna(inplace=True)
plot_throughput_cdf(data)
