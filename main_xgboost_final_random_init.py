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

os.chdir('/Users/farismismar/Desktop/DeepMIMO')

# 0) Some parameters
seed = 0
K_fold = 2
learning_rate = 0.01
max_users = 54481
r_exploitation = 0.4
p_blockage = 0.4

# in Mbps
# [3.8569 3.9503 4.0436 4.1370 4.2303 4.3237 4.4170 4.5104].
rate_threshold = 4.1

# in ms
gap_duration = 1
radio_frame_duration = 10

# in Watts
PTX_35 = 1 # in Watts for 3.5 GHz
PTX_28 = 1 # in Watts for 28 GHz

# speed:
v_s = 5 # km/h

delta_f_35 = 180e3 # Hz/subcarrier
delta_f_28 = 180e3 # Hz/subcarrier
N_SC_35 = 1
N_SC_28 = 1

mmWave_BW_multiplier = 1.39 # should be 10, but not currently.
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
    # Takes the two .csv files and merges them in a way that is useful for the Deep Learning.
    df35 = pd.read_csv('dataset/dataset_3.5_GHz.csv')
    df28_b = pd.read_csv('dataset/dataset_28_GHz_blockage.csv')
    df28_nb = pd.read_csv('dataset/dataset_28_GHz.csv')
    
    # Truncate to the first max_users rows, for efficiency for now
    df35 = df35.iloc[:max_users,:]
    df28_b = df28_b.iloc[:max_users,:]
    df28_nb = df28_nb.iloc[:max_users,:]
      
    assert(np.all(df28_b.iloc[:,513:516] == df28_nb.iloc[:,513:516]))
    
    # Based on blocking probability, create df28.
    p_b = np.random.binomial(1, p=p_blockage, size=max_users)
    df28 = df28_b.copy()
    df28.loc[(p_b==1),:] = df28_b.loc[(p_b == 1),:]
    df28.loc[(p_b==0),:] = df28_nb.loc[(p_b == 0),:]
    
    # Map: 0 is ID; 1-256 are H real; 257-512 are Himag; 513-515 are x,y,z 
    # 2) Perform data wrangling and construct the proper channel matrix H
    H35_real = df35.iloc[:,1:256+1]
    H35_imag = df35.iloc[:,257:512+1]
    H35_loc = df35.iloc[:,513:]
        
    H28_real = df28.iloc[:,1:256+1]
    H28_imag = df28.iloc[:,257:512+1] 
    H28_loc = df28.iloc[:,513:]

    # 2.5) We decided to slash the 3.5 Matrix to a UPA of 8x8 in the y-z plane.
    # Following the procedure
    H35_real_8 = []
    H35_imag_8 = []
    
    for user_id in np.arange(max_users):
        H35_real_i = H35_real.iloc[user_id, 0:256]
        H35_real_i = np.array(H35_real_i).reshape(32,8).T
        H35_real_i = H35_real_i.flatten()
        H35_real_i = H35_real_i[0:64]
        H35_imag_i = H35_imag.iloc[user_id, 0:256]
        H35_imag_i = np.array(H35_imag_i).reshape(32,8).T
        H35_imag_i = H35_imag_i.flatten()
        H35_imag_i = H35_imag_i[0:64]
        
        H35_real_8.append(H35_real_i)
        H35_imag_8.append(H35_imag_i)
        
    # Convert to pandas df    
    H35_real_8 = pd.DataFrame(H35_real_8)
    H35_imag_8 = pd.DataFrame(H35_imag_8)
    
    # Before moving forward, check if the loc at time t is equal
    assert(np.all(df35.iloc[:,513:516] == df28_b.iloc[:,513:516]))
    
    # Reset the column names of the imaginary H
    H35_imag.columns = H35_real.columns
    H28_imag.columns = H28_real.columns
    
    H35 = H35_real + 1j * H35_imag
    H28 = H28_real + 1j * H28_imag
    
    del H35_real, H35_imag, H35_loc, H28_real, H28_imag, H28_loc
    
    channel_gain_35 = []
    channel_gain_28 = []
    
    # Compute the channel gain |h|^2
    for i in np.arange(max_users):
        H35_i = np.array(H35.iloc[i,:])
        H28_i = np.array(H28.iloc[i,:])
        channel_gain_35.append(np.vdot(H35_i, H35_i))
        channel_gain_28.append(np.vdot(H28_i, H28_i))
    
    # 3) Feature engineering: introduce RSRP mmWave and sub-6 and y
    channel_gain_28 = np.array(channel_gain_28).astype(float)
    channel_gain_35 = np.array(channel_gain_35).astype(float)
    
    # Get rid of unwanted columns in 3.5
    df35 = df35[['0', '513', '514', '515']]
    df35.columns = ['user_id', 'lon', 'lat', 'height']
    df35 = pd.concat([df35, H35_real_8, H35_imag_8], axis=1)
    
    df35.loc[:,'P_RX_35'] = 10*np.log10(PTX_35 * 1e3 * 4 * channel_gain_35) # since we slashed 256 to 64, we have 4 as extra gain.
    df28.loc[:,'P_RX_28'] = 10*np.log10(PTX_28 * 1e3 * channel_gain_28) 
    
    # These columns are redundant
    df28.drop(['0', '513', '514', '515'], axis=1, inplace=True)
    df = pd.concat([df35, df28], axis=1)
    
    df = df.iloc[:max_users,:]
    df = df[['user_id', 'lon', 'lat', 'height', 'P_RX_35', 'P_RX_28']]
    df.to_csv('dataset.csv', index=False)
    
    return df

def plot_confusion_matrix(y_test, y_pred, y_score):
    # Compute confusion matrix
    classes = [0,1]
    class_names = ['Deny','Grant']
    normalize = False
    
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    plt.figure(figsize=(8,5))
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks, class_names)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.tight_layout()
    plt.savefig('figures/conf_matrix.pdf', format='pdf')

def generate_roc(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)

    roc_auc_score_value = roc_auc_score(y_test, y_score)
#    print('The ROC AUC for this UE is {0:.6f}'.format(roc_auc_score_value))

    return fpr, tpr, roc_auc_score_value 

def plot_roc(fpr, tpr, roc_auc, i=0):
    plt.figure(figsize=(8,5))
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']   

    lw = 2
    
    plt.plot(fpr, tpr,
         lw=lw, label="ROC curve (AUC = {:.6f})".format(roc_auc))
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
#    plt.title(r'\textbf{Receiver Operating Characteristic -- UE \#' + '{0}'.format(i) + '}')
    plt.legend(loc="lower right")
    plt.savefig('figures/roc_{0}.pdf'.format(i), format='pdf')
    
def plot_throughput_cdf(T):
    fig = plt.figure(figsize=(10.24, 7.68))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 20
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
    plt.savefig('figures/throughputs.pdf', format='pdf')
    
def plot_primary(X,Y, title, xlabel, ylabel, filename='plot.pdf'):
    fig = plt.figure(figsize=(10.24,7.68))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssymb}']
    
    #plt.title(title)
    plt.xlabel(xlabel)
    
    ax = fig.gca()
    ax.set_autoscaley_on(True)
    
    plot_, = ax.plot(X, Y, 'k^-') #, label='ROC')

#    ax.set_xlim(xmin=0.15, xmax=0.55)
    
    ax.set_ylabel(ylabel)
#    ax.set_ylim(0.99, 1.01)
    
    plt.grid(True)
    fig.tight_layout()
    plt.savefig('figures/{}'.format(filename), format='pdf')
    plt.show()

def train_classifier(df, r_training=0.8):
    dataset = df.copy()
    
#    N_training = int(r_training * dataset.shape[0])
#    training = dataset.iloc[:N_training,:]
#    test = dataset.iloc[N_training:,:]
    
    training, test = train_test_split(dataset, train_size=r_training, random_state=seed)
    
    eps = 1e-9
    X_train = training.drop('y', axis=1)
    y_train = training['y']
    X_test = test.drop('y', axis=1)
    y_test = test['y']

    w = len(y_train[y_train == 0]) / (eps + len(y_train[y_train == 1]))
    
    print('Positive class weight: {}'.format(w))
    
    classifier = xgb.XGBClassifier(seed=seed, learning_rate=0.05, n_estimators=1000, max_depth=8, scale_pos_weight=w, silent=True)
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

def predict_handover(df, clf):
    y_test = df['y']
    X_test = df.drop(['y'], axis=1)
    
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    
    try:
        # Compute area under ROC curve
        roc_auc = roc_auc_score(y_test, y_score[:,1])
        print('The ROC AUC for this UE in the exploitation period is {:.6f}'.format(roc_auc))
    
        # Save the value
        f = open("figures/output.txt", 'a')
        f.write('ROC exploitation: {0},{1:.3f}\n'.format(r_exploitation, roc_auc))
        f.close()

        y_pred=pd.DataFrame(y_pred)
      
    except:
       print('The ROC AUC for this UE in the exploitation period is N/A')
       y_pred = None
       
    return y_pred

def get_beam_training_time(df, freq=28e9, horiz_beams=32):
    return 10e-3 * horiz_beams # 10 us in ms per beam.

def get_coherence_time(df, freq):
    c = 299792458 # m/s
    BS_x, BS_y, BS_z = [235.504198, 489.503816, 6]
    np.random.seed(seed)
    
    # Check if freq is mmWave 
    # then the beam coherence time
    # else 
    # OFDM coherence
    # Constant for all users
    
    # Obtain D
    # alpha AoA equivalent random(0, pi) or 30 to 150 degrees
    if (freq > 20e9): # mm-Wave
        D = ((df['lon'] - BS_x) ** 2 + (df['lat'] - BS_y) ** 2 + (df['height'] - BS_z) ** 2) ** 0.5
        Theta_n = 1.772 / math.sqrt(32) #112 / 32. # 32 antennas in the aziumuth direction # 3 dB BW of antenna
        alpha = np.random.uniform(0, math.pi, size=df.shape[0])
        T_B = D / (v_s * 1000/3600 * np.sin(alpha)) * Theta_n / 2.
        T = np.array(T_B).mean() * 1e3 # in ms
        print('INFO: Average coherence time for mmWave is {} ms'.format(T))
        return T
    
    if (freq < 20e9): # sub-6
        T = c / (freq * v_s * 1000/3600) * 1e3 
        print('INFO: Coherence time for sub-6 is {} ms'.format(T))
        return T * np.ones(df.shape[0]) # in ms

#df_ = create_dataset() # only uncomment for the first run, when the channel consideration changed.
df_ = pd.read_csv('dataset.csv')

# Dataset column names
# 0 user ID
# 1-3  (x,y,z) of user ID
# 4-67 real H35
# 68-131 imag 35
# 132 receive power 3.5 GHz in dBm
# 133-388 real H28
# 389-644 imag H28
# 645 receive power 28 GHz in dBm

df = df_.iloc[:max_users,:]
del df_

# Feature engineering: add SNR to the computation:
noise_floor_35 = k_B * T * delta_f_35 * 1e3
noise_floor_28 = k_B * T * delta_f_28 * 1e3 # in mW

noise_power_35 = 10 ** (Nf/10.) * noise_floor_35
noise_power_28 = 10 ** (Nf/10.) * noise_floor_28 

df['Capacity_35'] = B_35*np.log2(1 + 10**(df['P_RX_35']/10.) / noise_power_35) / 1e6
df['Capacity_28'] = B_28*np.log2(1 + 10**(df['P_RX_28']/10.) / noise_power_28) / 1e6

df = df[['lon', 'lat', 'height', 'Capacity_35', 'Capacity_28']]

# Compute the Effective Achievable Rates
coherence_time_sub6 = get_coherence_time(df, freq=3.5e9)
coherence_time_mmWave = get_coherence_time(df, freq=28e9) 
beam_training_penalty_mmWave = get_beam_training_time(df, freq=28e9)

df['Capacity_28'] *= 1 - 2 * beam_training_penalty_mmWave / coherence_time_mmWave
# TODO what is the formula for 3.5?

# ----------------------------------------------------------------------------
# TODO: Problem, initialize UEs randomly between 3.5 and 28 GHz (target)
# ----------------------------------------------------------------------------
df['Source'] = df['Capacity_35'].copy()
df['Target'] = df['Capacity_28'].copy()

##############################################################################
# The HO criterion
df['y'] = pd.DataFrame((df.loc[:,'Source'] <= rate_threshold) & (df.loc[:,'Target'] >= df.loc[:,'Source']), dtype=int)
df['Source_is_3.5'] = (df['Source'] == df['Capacity_35']) + 0
df['Source_is_28'] = (df['Source'] == df['Capacity_28']) + 0

# Change the order of columns to put 
column_order = ['lon', 'lat', 'height', 'Source', 'Target', 'Source_is_3.5', 'Source_is_28', 'y']
df = df[column_order]

##############################################################################
# 1) Optimal algorithm
##############################################################################
df_optimal = df.copy()

# No handover, so the throughput is the throughput of the source.
df_optimal.loc[df_optimal['y'] == 0, 'Capacity_Optimal'] = 1. * df_optimal.loc[(df_optimal['y'] == 0), 'Source']

# Handover choose the maximum of both rates.
df_optimal.loc[df_optimal['y'] == 1, 'Capacity_Optimal'] = df_optimal.loc[df_optimal['y'] == 1,['Source', 'Target']].apply(np.max, axis=1)

# Sample r_exploit data randomly from df_optimal
benchmark_data_optimal = df_optimal.iloc[np.random.randint(low=0, high=df_optimal.shape[0], size=N_exploit), :]

del df_optimal

##############################################################################
# 2) Legacy algorithm
##############################################################################
df_legacy = df.copy()

# Penalize the throughput rates aka Effective Achievable Rate
weight_source_sub6 = (radio_frame_duration - gap_duration) / radio_frame_duration
weight_source_mmWave = (radio_frame_duration - gap_duration) / radio_frame_duration

# No handover, so the throughput is the throughput of the source.
df_legacy.loc[df_legacy['y'] == 0, 'Capacity_Legacy'] = 1. * df_legacy.loc[(df_legacy['y'] == 0), 'Source']

# Handover chooses the weighted average between the source and target.
agg_rate_ho_from35 = weight_source_sub6 * df_legacy.loc[(df_legacy['y'] == 1) & (df_legacy['Source_is_3.5'] == 1), 'Source'] + (1 - weight_source_sub6) * df_legacy.loc[(df_legacy['y'] == 1) & (df_legacy['Source_is_3.5'] == 1), 'Target']  # from 3.5 to mmWave
agg_rate_ho_from28 = weight_source_mmWave * df_legacy.loc[(df_legacy['y'] == 1) & (df_legacy['Source_is_28'] == 1), 'Source'] + (1 - weight_source_mmWave) * df_legacy.loc[(df_legacy['y'] == 1) & (df_legacy['Source_is_28'] == 1), 'Target'] # from mmWave to 3.5

df_legacy.loc[df_legacy['y'] == 1, 'Capacity_Legacy'] = pd.concat([agg_rate_ho_from35,agg_rate_ho_from28]) # this concat will preserve the index.
##

# Sample r_exploit data randomly from df_legacy
benchmark_data_legacy = df_legacy.iloc[np.random.randint(low=0, high=df_legacy.shape[0], size=N_exploit), :]

del df_legacy

##############################################################################
# 3) Blind handover algorithm
##############################################################################

df_blind = df.copy()

df_blind['y'] = pd.DataFrame((df_blind.loc[:,'Source'] <= rate_threshold), dtype=int)

# no gap for blind handover
weight_source_sub6 = (radio_frame_duration - 0) / radio_frame_duration
weight_source_mmWave = (radio_frame_duration - 0) / radio_frame_duration

# No handover, so the throughput is the throughput of the source.
df_blind.loc[df_blind['y'] == 0, 'Capacity_Blind'] = 1. * df_blind.loc[(df_blind['y'] == 0), 'Source']

# Handover for blind 
agg_rate_ho_from35 = weight_source_sub6 * df_blind.loc[(df_blind['y'] == 1) & (df_blind['Source_is_3.5'] == 1), 'Target'] + (1 - weight_source_sub6) * df_blind.loc[(df_blind['y'] == 1) & (df_blind['Source_is_3.5'] == 1), 'Source']  # from 3.5 to mmWave
agg_rate_ho_from28 = weight_source_mmWave * df_blind.loc[(df_blind['y'] == 1) & (df_blind['Source_is_28'] == 1), 'Target'] + (1 - weight_source_mmWave) * df_blind.loc[(df_blind['y'] == 1) & (df_blind['Source_is_28'] == 1), 'Source'] # from mmWave to 3.5

df_blind.loc[df_blind['y'] == 1, 'Capacity_Blind'] = pd.concat([agg_rate_ho_from35,agg_rate_ho_from28])  # this concat will preserve the index.
##

# Sample r_exploit data randomly from df_blind
benchmark_data_blind = df_blind.iloc[np.random.randint(low=0, high=df_blind.shape[0], size=N_exploit), :]

del df_blind

##############################################################################
# 4) Proposed algorithm
##############################################################################

# The height column must be deleted here before prediction is made
height = df['height']
df_proposed = df.drop(['height'], axis=1)

# Use this for the exploitation
train_valid, benchmark_data_proposed = train_test_split(df_proposed, test_size=r_exploitation, random_state=seed)
    
roc_graphs = pd.DataFrame()
roc_auc_values = []

# Change r_training and save roc1 then repeat
max_r_training = 0
max_score = 0
best_clf = None
X = np.arange(1,10,1)/10.
for r_t in X:
    try:
        [y_pred, y_score, clf] = train_classifier(train_valid, r_t)
        y_pred_proposed = predict_handover(benchmark_data_proposed, clf)
        y_score_proposed = clf.predict_proba(benchmark_data_proposed.drop(['y'], axis=1))
        y_test_proposed = benchmark_data_proposed['y']

        fpr, tpr, score = generate_roc(y_test_proposed, y_score_proposed[:,1])
        if (score > max_score):
            max_score = score
            max_r_training = r_t
            best_clf = clf
            
        roc_auc_values.append(score)
        
        roc_graphs = pd.concat([roc_graphs, pd.DataFrame(fpr), pd.DataFrame(tpr)], axis=1)
    except:
        roc_auc_values.append(np.nan)
        pass

# Replace all NaNs with 1.00000 since they are coming at the end
roc_graphs = roc_graphs.fillna(1)
roc_graphs.to_csv('roc_output.csv', index=False)
plot_primary(X, roc_auc_values, 'ROC vs Training', r'$r_\text{training}$', 'ROC AUC', filename='roc_vs_training.pdf')

# Now generate data with the best classifier.
y_pred_proposed = predict_handover(benchmark_data_proposed, best_clf)
y_score_proposed = best_clf.predict_proba(benchmark_data_proposed.drop(['y'], axis=1))
y_test_proposed = benchmark_data_proposed['y']

plot_confusion_matrix(y_test_proposed, y_pred_proposed, y_score_proposed)

# Put back the height column
benchmark_data_proposed['height'] = height

# Penalize the throughput rates aka Effective Achievable Rate
# We also eliminate the measurement gap here, and use the legacy formula
weight_source_sub6 = (radio_frame_duration - 0) / radio_frame_duration
weight_source_mmWave = (radio_frame_duration - 0) / radio_frame_duration

# No handover, so the throughput is the throughput of the source.
benchmark_data_proposed.loc[benchmark_data_proposed['y'] == 0, 'Capacity_Proposed'] = 1. * benchmark_data_proposed.loc[(benchmark_data_proposed['y'] == 0), 'Source']

agg_rate_ho_from35 = weight_source_sub6 * benchmark_data_proposed.loc[(benchmark_data_proposed['y'] == 1) & (benchmark_data_proposed['Source_is_3.5'] == 1), 'Source'] + (1 - weight_source_sub6) * benchmark_data_proposed.loc[(benchmark_data_proposed['y'] == 1) & (benchmark_data_proposed['Source_is_3.5'] == 1), 'Target']  # from 3.5 to mmWave
agg_rate_ho_from28 = weight_source_mmWave * benchmark_data_proposed.loc[(benchmark_data_proposed['y'] == 1) & (benchmark_data_proposed['Source_is_28'] == 1), 'Source'] + (1 - weight_source_mmWave) * benchmark_data_proposed.loc[(benchmark_data_proposed['y'] == 1) & (benchmark_data_proposed['Source_is_28'] == 1), 'Target'] # from mmWave to 3.5

benchmark_data_proposed.loc[benchmark_data_proposed['y'] == 1, 'Capacity_Proposed'] = pd.concat([agg_rate_ho_from35,agg_rate_ho_from28])  # this concat will preserve the index.
##

##############################################################################
# Plotting
##############################################################################
benchmark_data_optimal = benchmark_data_optimal.reset_index().drop(['index'], axis=1)
benchmark_data_proposed = benchmark_data_proposed.reset_index().drop(['index'], axis=1)
benchmark_data_legacy  = benchmark_data_legacy.reset_index().drop(['index'], axis=1)
benchmark_data_blind = benchmark_data_blind.reset_index().drop(['index'], axis=1)
benchmark_data_proposed = benchmark_data_proposed.reset_index().drop(['index'], axis=1)

# Temporary
benchmark_data_proposed['Capacity_35'] = benchmark_data_proposed['Source']
benchmark_data_proposed['Capacity_28'] = benchmark_data_proposed['Target']

data = pd.concat([benchmark_data_optimal['Capacity_Optimal'], benchmark_data_proposed['Capacity_Proposed'], benchmark_data_legacy['Capacity_Legacy'], benchmark_data_blind['Capacity_Blind'], benchmark_data_proposed['Capacity_35'], benchmark_data_proposed['Capacity_28']], axis=1, ignore_index=True)
data.columns = ['Optimal', 'Proposed', 'Legacy', 'Blind', 'Sub-6 only', 'mmWave only']
data.dropna(inplace=True)

data.to_csv('dataset_post.csv', index=False)
plot_throughput_cdf(data)