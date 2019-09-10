#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:08:03 2019
@author: farismismar
"""

import random
import os
import numpy as np
import pandas as pd
import math

import itertools
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.ticker import MultipleLocator, FuncFormatter

import matplotlib2tikz

os.chdir('/Users/farismismar/Desktop/DeepMIMO')
# 0) Some parameters
seed = 0
K_fold = 2
learning_rate = 0.05
max_users = 54481
r_exploitation = 0.8
p_blockage = 0.4

p_randomness = 0 # 0 = all users start in 3.5

# in Mbps
rate_threshold_sub6 = 1.72 # median
rate_threshold_mmWave = 7.00

training_request_handover_threshold = np.inf #(1 - p_randomness) * rate_threshold_sub6 + p_randomness * rate_threshold_mmWave  # this is x_hr, but only for the training data.
request_handover_threshold = (1 - p_randomness) * rate_threshold_sub6 + p_randomness * rate_threshold_mmWave  # this is x_hr

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

mmWave_BW_multiplier = 10 # x sub-6
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
   # classes = [0,1]
    class_names = ['Deny','Grant']
    normalize = False
    
    cm, _  = get_misclassification_error(y_test, y_pred, y_score)
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    fig = plt.figure(figsize=(10,9))
    ax = fig.gca()
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 40
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto', origin='lower')

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


def _parula_map():
    # https://stackoverflow.com/questions/34859628/has-someone-made-the-parula-colormap-in-matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    
    cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
     [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
     [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
      0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
     [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
      0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
     [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
      0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
     [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
      0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
     [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
      0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
     [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
      0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
      0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
     [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
      0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
     [0.0589714286, 0.6837571429, 0.7253857143], 
     [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
     [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
      0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
     [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
      0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
     [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
      0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
     [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
      0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
     [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
     [0.7184095238, 0.7411333333, 0.3904761905], 
     [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
      0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
     [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
     [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
      0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
     [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
      0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
     [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
     [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
     [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
      0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
     [0.9763, 0.9831, 0.0538]]
    
    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
    
    return parula_map

def plot_joint_pdf(X, Y):
    fig = plt.figure(figsize=(10.24, 7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 30
    matplotlib.rcParams['xtick.labelsize'] = 'small'
    matplotlib.rcParams['ytick.labelsize'] = 'small'
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']   
    
    num_bins = 50
    H, X_bin_edges, Y_bin_edges = np.histogram2d(X, Y, bins=(num_bins, num_bins), normed=True)
    for y in np.arange(num_bins):
        H[y,:] = H[y,:] / sum(H[y,:])
    pdf = H / num_bins    
    
    ax = plt.gca(projection="3d")
    
    x, y = np.meshgrid(X_bin_edges, Y_bin_edges)

    surf = ax.plot_surface(x[:num_bins, :num_bins], y[:num_bins, :num_bins], pdf[:num_bins, :num_bins], cmap=_parula_map(), antialiased=True)
    #cb = fig.colorbar(surf, shrink=0.5)
    ax.view_init(5, 45) # the first param rotates the z axis inwards or outwards the screen.  The second is our guy.
    
    # No background color    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.set_xlabel('3.5 GHz')
    ax.set_ylabel('28 GHz')
    ax.set_zlabel('Joint Throughput pdf')

    ax.invert_xaxis()
    ax.invert_yaxis()
        
    ax.set_xlim(int(np.max(X)), 0)
    ax.set_ylim(int(np.max(Y)), 0)
    ax.set_zlim(np.min(pdf), np.max(pdf))
        
    ax.xaxis.labelpad=20
    ax.yaxis.labelpad=20
    ax.zaxis.labelpad=20
    
    plt.xticks([3,2,1,0])
    plt.yticks([15,10,5,0])
        
    plt.tight_layout()
    
    plt.savefig('figures/joint_throughput_pdf_{}.pdf'.format(p_randomness), format='pdf')
    matplotlib2tikz.save('figures/joint_throughput_pdf_{}.tikz'.format(p_randomness))

def plot_joint_cdf(X, Y):
    fig = plt.figure(figsize=(10.24, 7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 30
    matplotlib.rcParams['xtick.labelsize'] = 'small'
    matplotlib.rcParams['ytick.labelsize'] = 'small'
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']   
    
    num_bins = 100
    H, X_bin_edges, Y_bin_edges = np.histogram2d(X, Y, bins=(num_bins, num_bins), normed=True)
    for y in np.arange(num_bins):
        H[y,:] = H[y,:] / sum(H[y,:])
    pdf = H / num_bins
    
    cdf = np.zeros((num_bins, num_bins))
    for i in np.arange(num_bins):
        for j in np.arange(num_bins):
            cdf[i,j] = sum(sum(pdf[:(i+1), :(j+1)]))

    ax = plt.gca(projection="3d")
    x, y = np.meshgrid(X_bin_edges, Y_bin_edges)

    surf = ax.plot_surface(x[:num_bins, :num_bins], y[:num_bins, :num_bins], cdf[:num_bins, :num_bins], cmap=_parula_map(), antialiased=True)
#    cb = fig.colorbar(surf, shrink=0.5)
    ax.view_init(5, 45) # the first param rotates the z axis inwards or outwards the screen.  The second is our guy.
    
    # No background color    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.set_xlabel('3.5 GHz')
    ax.set_ylabel('28 GHz')
    ax.set_zlabel('Joint Throughput CDF')
    
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    ax.set_xlim(int(np.max(X)), 0)
    ax.set_ylim(int(np.max(Y)), 0)
    ax.set_zlim(0,1)

    ax.xaxis.labelpad=20
    ax.yaxis.labelpad=20
    ax.zaxis.labelpad=20

    plt.xticks([3,2,1,0])
    plt.yticks([15,10,5,0])
    
    plt.tight_layout()
    
    plt.savefig('figures/joint_throughput_cdf_{}.pdf'.format(p_randomness), format='pdf')
    matplotlib2tikz.save('figures/joint_throughput_cdf_{}.tikz'.format(p_randomness))

def plot_pdf(data1, label1, data2, label2):
    fig = plt.figure(figsize=(10.24, 7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 40
    matplotlib.rcParams['xtick.labelsize'] = 'small'
    matplotlib.rcParams['ytick.labelsize'] = 'small'
    matplotlib.rcParams['legend.fontsize'] =  'small'
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
    
def plot_throughput_cdf(T, filename, legend=True):
    fig = plt.figure(figsize=(10.24, 7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 40
    matplotlib.rcParams['xtick.labelsize'] = 'small'
    matplotlib.rcParams['ytick.labelsize'] = 'small'
    matplotlib.rcParams['legend.fontsize'] =  'smaller'
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']   
    
    labels = T.columns

    num_bins = 50

    for data in T:
        data_ = T[data]

        counts, bin_edges = np.histogram(data_, bins=num_bins, density=True)
        cdf = np.cumsum(counts) / counts.sum()
        ax = fig.gca()
        if data == 'mmWave only':
            style = 'r-'
        elif data == 'Sub-6 only':
            style = 'b-'
        elif data == 'Optimal':
            style = '^--'
        elif data == 'Proposed':
#            lw = 3.5
            style = '+-'
        else:
            style = '-'
        ax.plot(bin_edges[1:], cdf, style, linewidth=2, markevery=10)

    plt.legend(labels, loc="best")
    
    if not legend:
        ax.get_legend().remove()
    
    plt.grid('both', linestyle='dashed')
    ax.set_ylim(0, 1)
    plt.xlabel('Throughput [Mbps]')
    plt.ylabel('Throughput CDF')
    plt.tight_layout()    
    
    plt.savefig('figures/{}.pdf'.format(filename), format='pdf')
    matplotlib2tikz.save('figures/{}.tikz'.format(filename))

def plot_throughput_pdf(T):
    fig = plt.figure(figsize=(10.24, 7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 40
    matplotlib.rcParams['xtick.labelsize'] = 'small'
    matplotlib.rcParams['ytick.labelsize'] = 'small'
    matplotlib.rcParams['legend.fontsize'] =  'small'
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
    plt.xlabel('Throughput [Mbps]')
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
    matplotlib.rcParams['xtick.labelsize'] = 'small'
    matplotlib.rcParams['ytick.labelsize'] = 'small'
    matplotlib.rcParams['legend.fontsize'] =  'small'
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
def train_classifier(df, r_training=0.8):
    dataset = df.copy()
    
    training, test = train_test_split(dataset, train_size=r_training, random_state=seed)
    
    eps = 1e-9
    X_train = training.drop('y', axis=1)
    y_train = training['y']
    X_test = test.drop('y', axis=1)
    y_test = test['y']

    w = len(y_train[y_train == 0]) / (eps + len(y_train[y_train == 1]))
    
    print('Positive class weight: {}'.format(w))
    
    classifier = xgb.XGBClassifier(seed=seed, learning_rate=learning_rate, n_estimators=1000, max_depth=8, scale_pos_weight=w, silent=True)
    #classifier.get_params().keys()
    
    # Hyperparameters
    alphas = np.linspace(0,1,2)
    lambdas = np.linspace(0,1,2)
    sample_weights = [0.5, 0.7]
    child_weights = [0, 10]
    objectives = ['binary:logistic']
    gammas = [0, 0.02, 0.04]
    
    hyperparameters = {'reg_alpha': alphas, 'reg_lambda': lambdas, 'objective': objectives, 
                       'colsample_bytree': sample_weights, 'min_child_weight': child_weights, 'gamma': gammas}
  
    gs_xgb = GridSearchCV(classifier, hyperparameters, scoring='roc_auc', cv=K_fold) # k-fold crossvalidation
    gs_xgb.fit(X_train, y_train)
    clf = gs_xgb.best_estimator_
    
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)

    try:
        roc_auc = roc_auc_score(y_test, y_score[:,1])
        print('The Training ROC AUC for this classifier is {:.6f}'.format(roc_auc))
    except:
        print('The Training ROC AUC for this classifier is N/A')

    return [y_pred, y_score, clf]

def predict_handover(df, clf, r_training):
    y_test = df['y']
    X_test = df.drop(['y'], axis=1)
    
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    
    try:
        # Compute area under ROC curve
        roc_auc = roc_auc_score(y_test, y_score[:,1])
        print('The ROC AUC for this UE in the exploitation period is {:.6f}'.format(roc_auc))
    
        # Save the value
        f = open("figures/output_xgboost_{}.txt".format(p_randomness), 'a')
        f.write('r_exploitation {0}, r_training {1}, ROC {2:.6f}\n'.format(r_exploitation, r_training, roc_auc))
        f.close()

        y_pred=pd.DataFrame(y_pred)
      
    except:
       print('The ROC AUC for this UE in the exploitation period is N/A')
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

#df_ = create_dataset() # only uncomment for the first run, when the channel consideration changes.  Otherwise, no need.
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

# Based on x_hr, if there was no handover, put the source effective rates back
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

# Based on x_hr, if there was no handover, put the source effective rates back
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

df_proposed.loc[:, 'HO_requested'] = (df_proposed.loc[:, 'Source'] < request_handover_threshold) + 0
df_proposed.loc[:, 'y'] = (df_proposed.loc[:,'Target'] >= df_proposed.loc[:,'Source']) + 0

# No handover request means no handover granted
df_proposed.loc[df_proposed['HO_requested'] == 0, 'y'] = 0

if (p_randomness == 0 or p_randomness == 1):
    df_proposed = df_proposed.drop(['Source_is_3.5'], axis=1) # these two values will make the column of a single value.

# Use this for the exploitation
train_valid, benchmark_data_proposed = train_test_split(df_proposed, test_size=r_exploitation, random_state=seed)

# The training and validation data get the infinity threshold (always request).
train_valid['HO_requested'] = 1

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
X = [1e-3,5e-3,7e-3,1e-2,3e-2,5e-2,7e-2,1e-1,3e-1,0.4,5e-1,7e-1] # note we removed 3e-3.
for r_t in X:
    try:
        [y_pred, y_score, clf] = train_classifier(train_valid, r_t)
        y_pred_proposed, score = predict_handover(benchmark_data_proposed, clf, r_t)
        y_score_proposed = clf.predict_proba(benchmark_data_proposed.drop(['y'], axis=1))
        y_test_proposed = benchmark_data_proposed['y']
        _, mu = get_misclassification_error(y_test_proposed, y_pred_proposed, y_score_proposed)

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

# Based on x_hr, if there was no handover, put the source effective rates back
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
benchmark_data_legacy  = benchmark_data_legacy.reset_index().drop(['index'], axis=1)
benchmark_data_blind = benchmark_data_blind.reset_index().drop(['index'], axis=1)
benchmark_data_proposed = benchmark_data_proposed.reset_index().drop(['index'], axis=1)
sub_6_capacities = sub_6_capacities.reset_index().drop(['index'], axis=1)
mmWave_capacities = mmWave_capacities.reset_index().drop(['index'], axis=1)

benchmark_data_proposed.loc[:,'y_true'] = benchmark_data_proposed['y'].copy()
benchmark_data_proposed['y'] = y_pred_proposed

# Summaries
f = open('figures/handover_metrics_{}.txt'.format(p_randomness), 'w')
for policy in ['proposed', 'legacy', 'blind']:
    d_ = eval('benchmark_data_{}'.format(policy))
    f.write('Policy {0} -- number of handovers requested in exploitation phase: {1:.0f}\n'.format(policy, d_['HO_requested'].sum()))
    f.write('Policy {0} -- number of handovers granted in exploitation phase: {1:.0f}\n'.format(policy, d_['y'].sum()))
f.close()
    
data = pd.concat([benchmark_data_optimal['Capacity_Optimal'], benchmark_data_proposed['Capacity_Proposed'], benchmark_data_proposed['HO_requested'], benchmark_data_legacy['Capacity_Legacy'], benchmark_data_blind['Capacity_Blind'], sub_6_capacities['Capacity_35'], mmWave_capacities['Capacity_28']], axis=1, ignore_index=True)
data.columns = ['Optimal', 'Proposed', 'HO_requested', 'Legacy', 'Blind', 'Sub-6 only', 'mmWave only']
data.to_csv('figures/dataset_post_{}.csv'.format(p_randomness), index=False)

#plot_throughput_pdf(data)
plot_throughput_cdf(data[['Sub-6 only', 'mmWave only']], 'throughput_cdf_{}'.format(p_randomness))

diff = pd.DataFrame(data = (abs(data['mmWave only'] - data['Sub-6 only'])), columns=['Difference'])
plot_throughput_cdf(diff, 'diff_cdf_{}'.format(p_randomness), legend=False)

# 3D Plot pdf/CDF
plot_joint_pdf(data['Sub-6 only'], data['mmWave only'])
plot_joint_cdf(data['Sub-6 only'], data['mmWave only'])

data_policies = data[['Optimal', 'Proposed', 'Legacy', 'Blind']]
data_policies.dropna(inplace=True)
plot_throughput_cdf(data_policies, 'throughputs_{}'.format(p_randomness))
