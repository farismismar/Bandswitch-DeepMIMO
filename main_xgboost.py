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
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.ticker import MultipleLocator, FuncFormatter

os.chdir('/Users/farismismar/Desktop/DeepMIMO')

# 0) Some parameters
seed = 0
K_fold = 3
learning_rate = 0.01
max_users = 54481
r_exploitation = 0.4

# in Mbps
rate_threshold_35 = 4.3
rate_threshold_28 = 2.9

# in ms
gap_duration = 1
frame_duration = 10

# in Watts
PTX_35 = 1 # in Watts for 3.5 GHz
PTX_28 = 1 # in Watts for 28 GHz

# speed:
v_s = 5 # km/h

delta_f_35 = 180e3 # Hz/subcarrier
delta_f_28 = 180e3 # Hz/subcarrier
N_SC_35 = 1
N_SC_28 = 1

mmWave_BW_multiplier = 1.2
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
    df28 = pd.read_csv('dataset/dataset_28_GHz_blockage.csv')
    
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
    assert(np.all(df35.iloc[:,513:516] == df28.iloc[:,513:516]))
    
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
        channel_gain_35.append(PTX_35 * np.vdot(H35_i, H35_i))
        channel_gain_28.append(PTX_28 * np.vdot(H28_i, H28_i))
    
    # 3) Feature engineering: introduce RSRP mmWave and sub-6 and y
    channel_gain_28 = np.array(channel_gain_28).astype(float)
    channel_gain_35 = np.array(channel_gain_35).astype(float)
    
    # introduce blockage as Bernoulli random variable
    #p = np.randon.binomial(1, p=p_blockage)
    #channel_gain_28 *= 10 ** (p * blockage_loss/10.)
        
    noise_floor_35 = k_B * T * delta_f_35
    noise_floor_28 = k_B * T * delta_f_28 * mmWave_BW_multiplier
    
    noise_power_35 = 10 ** (Nf/10.) * noise_floor_35
    noise_power_28 = 10 ** (Nf/10.) * noise_floor_28
    
    # Get rid of unwanted columns in 3.5
    df35 = df35[['0', '513', '514', '515']]
    df35.columns = ['t', 'lon', 'lat', 'height']
    df35 = pd.concat([df35, H35_real_8, H35_imag_8], axis=1)
    
    df35.loc[:,'Capacity_35'] = B_35 * np.log2(1 + PTX_35 * 4 * channel_gain_35 / noise_power_35) / 1e6
    df28.loc[:,'Capacity_28'] = B_28 * np.log2(1 + PTX_28 * channel_gain_28 / noise_power_28) / 1e6
    
    # These columns are redundant
    df28.drop(['0', '513', '514', '515'], axis=1, inplace=True)
    df = pd.concat([df35, df28], axis=1)
    
    df.to_csv('dataset.csv', index=False)

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
    roc_auc = auc(fpr, tpr)

    roc_auc_score_value = roc_auc_score(y_test, y_score)
    print('The ROC AUC for this UE is {0:.6f}'.format(roc_auc_score_value))

    return fpr, tpr, roc_auc, roc_auc_score 

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

    num_bins =  50
    i = 0
    for data in T:
        data_ = T[data]

        counts, bin_edges = np.histogram(data_, bins=num_bins, density=True)
        cdf = np.cumsum(counts) / counts.sum()
        lw = 1 + i
        ax = fig.gca()
        if data == 'Optimal':
            style = '--'
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
        print('The Training ROC AUC for this UE is {:.6f}'.format(roc_auc))
    except:
        print('The Training ROC AUC for this UE is N/A')

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

def get_coherence_time(df, freq):
    c = 299792458 # m/s
    BS_x, BS_y, BS_z = [235.504198, 489.503816, 6]

    # Check if freq is mmWave 
    # then the beam coherence time
    # else 
    # OFDM coherence
    # Constant for all users
    
    # Obtain D
    # alpha AoA equivalent random(0, pi) or 30 to 150 degrees
    if (freq > 20e9): # mm-Wave
        D = ((df['lon'] - BS_x) ** 2 + (df['lat'] - BS_y) ** 2 + (df['height'] - BS_z) ** 2) ** 0.5
        Theta_n = 112 / 32. # 32 antennas in the aziumuth direction
        alpha = np.random.uniform(0, math.pi, size=df.shape[0])
        T_B = D / (v_s * 1e3 * np.sin(alpha)) * Theta_n / 2. # in ms.
        
        return np.array(T_B).mean()
    
    if (freq < 20e9): # sub-6
        return c / (freq * v_s * 1e3) * 1e3 # in ms

#create_dataset() # only uncomment for the first run.
df_ = pd.read_csv('dataset.csv')

# Dataset column names
# 0 user ID
# 1-3  (x,y,z) of user ID
# 4-67 real H35
# 68-131 imag 35
# 132 capacity rate_35 in MHz
# 133-388 real H28
# 389-644 imag H28
# 645 capacity_rate_28 in MHz

df = df_.iloc[:max_users,:]
del df_

df = df[['Capacity_35', 'Capacity_28', 'lon', 'lat', 'height']]

# TODO: Later to see how this works
# Randomly assign users to 28 GHz and 3.5 GHz
#df.loc[:, 'Camped3.5'] = np.random.binomial(1, 0.5, df.shape[0])
#df.loc[:, 'Camped28'] = 1 - df.loc[:, 'Camped3.5']

# َQ: Is there a correlation between the channels?

# ----------------------------------------------------------------------------
# Problem I am in 3.5 GHz (src) and want to HO to 28 GHz (target)
# ----------------------------------------------------------------------------

# Drop the NA data (a single unwanted row)
#df.dropna(inplace=True)

# Define the HO criterion here.
df['y'] = pd.DataFrame((df.loc[:,'Capacity_35'] <= rate_threshold_35) & (df.loc[:,'Capacity_28'] >= rate_threshold_28), dtype=int)

# Change the order of columns to put 
#column_order = ['lon', 'lat', 'height', 'Capacity_35', 'Capacity_28', 'Capacity_28_t+1', 'Capacity_28_dt+1', 'y']
column_order = ['lon', 'lat', 'height', 'Capacity_35', 'Capacity_28', 'y']
df = df[column_order]

##############################################################################
# 1) Optimal algorithm
##############################################################################
df_optimal = df.copy()
df_optimal.loc[:,'Capacity_Optimal'] = df_optimal[['Capacity_35', 'Capacity_28']].apply(np.max, axis=1)
benchmark_data_optimal = df_optimal.iloc[N_exploit:,:]
del df_optimal

##############################################################################
# 2) Legacy algorithm
##############################################################################
df_legacy = df.copy()
df_legacy['y'] = pd.DataFrame((df_legacy.loc[:,'Capacity_35'] <= rate_threshold_35) & (df_legacy.loc[:,'Capacity_28'] >= rate_threshold_28), dtype=int)

p_0 = gap_duration / frame_duration

tx_f1 = 1 - gap_duration / get_coherence_time(df_legacy, freq=3.5e9) # 3.5
tx_f2 = 1 / get_coherence_time(df_legacy, freq=28e9) # there is no measurement gap to the target frequency 28

agg_rate = tx_f1 / (tx_f1 + tx_f2) * df_legacy.loc[:,'Capacity_35'] + tx_f2 / (tx_f1 + tx_f2) * df_legacy.loc[:,'Capacity_28']
df_legacy.loc[:,'Capacity_Legacy'] = p_0 * 0 + (1 - p_0) * agg_rate

benchmark_data_legacy = df_legacy.iloc[N_exploit:,:]
del df_legacy

##############################################################################
# 3) Blind handover algorithm
##############################################################################
df_blind = df.copy()

df_blind['y'] = pd.DataFrame((df_blind.loc[:,'Capacity_35'] <= rate_threshold_35), dtype=int)
df_blind.loc[df_blind['y'] == 1, 'Capacity_Blind'] = df_blind['Capacity_28']
df_blind.loc[df_blind['y'] == 0, 'Capacity_Blind'] = df_blind['Capacity_35']

benchmark_data_blind = df_blind.iloc[N_exploit:,:]
del df_blind

##############################################################################
# 4) Proposed algorithm
##############################################################################

# The height column must be deleted here before prediction is made
df.drop(['height'], axis=1, inplace=True)

# Use this for the exploitation
train = df.iloc[:N_exploit,:]
benchmark_data_proposed = df.iloc[N_exploit:,:]

roc_graphs = pd.DataFrame()
roc_auc_values = []

# Change r_training and save roc1 then repeat
X = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for r_t in X:
    [y_pred, y_score, clf] = train_classifier(train, r_t)
    y_pred_proposed = predict_handover(benchmark_data_proposed, clf)
    y_score_proposed = clf.predict_proba(benchmark_data_proposed.drop(['y'], axis=1))
    y_test_proposed = benchmark_data_proposed['y']

    fpr, tpr, score, _ = generate_roc(y_test_proposed, y_score_proposed[:,1])
    roc_auc_values.append(score)
    
    roc_graphs = pd.concat([roc_graphs, pd.DataFrame(fpr), pd.DataFrame(tpr)], axis=1)

roc_graphs.to_csv('roc_output.csv', index=False)
plot_primary(X, roc_auc_values, 'ROC vs Training', r'$r_\text{training}$', 'ROC AUC', filename='roc_vs_training.pdf')

# Run these lines for the target r_train only.
plot_confusion_matrix(y_test_proposed, y_pred_proposed, y_score_proposed)
benchmark_data_proposed.loc[benchmark_data_proposed['y'] == 0, 'Capacity_Proposed'] = benchmark_data_proposed['Capacity_35']
benchmark_data_proposed.loc[benchmark_data_proposed['y'] == 1, 'Capacity_Proposed'] = benchmark_data_proposed['Capacity_28']

##############################################################################
# Plotting
##############################################################################
#data = pd.concat([benchmark_data_optimal['Capacity_Optimal'], benchmark_data_proposed['Capacity_Proposed'], benchmark_data_legacy['Capacity_Legacy'], benchmark_data_blind['Capacity_Blind'], benchmark_data_proposed['Capacity_35'], benchmark_data_proposed['Capacity_28']], axis=1)
#data.columns = ['Optimal', 'Proposed', 'Legacy', 'Blind', 'Sub-6 only', 'mmWave only']

data = pd.concat([benchmark_data_optimal['Capacity_Optimal'], benchmark_data_proposed['Capacity_Proposed'], benchmark_data_proposed['Capacity_35'], benchmark_data_proposed['Capacity_28']], axis=1)
data.columns = ['Optimal', 'Proposed', 'Sub-6 only', 'mmWave only']
plot_throughput_cdf(data)