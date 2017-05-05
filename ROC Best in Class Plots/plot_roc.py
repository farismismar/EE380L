# -*- coding: utf-8 -*-
"""
Created on Sun April 30 19:46:45 2017

@author: farismismar
"""
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib import rc

os.chdir('/Users/farismismar/Desktop/E_Projects/UT Austin Ph.D. EE/Courses/4- Spring 2017/Data Mining/Project/ROC Best in Class Plots')

dataset = 'ER'

lw = 2

plt.figure(figsize=(13,8))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')

################################################################################################
model = 'KNN'
sampling = 'SMOTE'
tpr = []
fpr = []
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr,
     lw=lw, label="ROC curve using {0} and {1} (AUC = {2:.6f})".format(model, sampling, roc_auc))
################################################################################################
model = 'NB'
sampling = 'Original'
tpr = []
fpr = []
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr,
     lw=lw, label="ROC curve using {0} and {1} (AUC = {2:.6f})".format(model, sampling, roc_auc))
################################################################################################
model = 'MLP'
sampling = 'Under-sampling 10\%'
tpr = []
fpr = []
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr,
     lw=lw, label="ROC curve using {0} and {1} (AUC = {2:.6f})".format(model, sampling, roc_auc))
################################################################################################
model = 'XGB'
sampling = 'Under-sampling 10\%'
tpr = []
fpr = []
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr,
     lw=lw, label="ROC curve using {0} and {1} (AUC = {2:.6f})".format(model, sampling, roc_auc))

################################################################################################
model = 'RFC'
sampling = 'Under-sampling 50\%'
tpr = []
fpr = []
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr,
     lw=lw, label="ROC curve using {0} and {1} (AUC = {2:.6f})".format(model, sampling, roc_auc))
################################################################################################
model = 'LR'
sampling = 'Under-sampling 50\%'
tpr = []
fpr = []
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr,
     lw=lw, label="ROC curve using {0} and {1} (AUC = {2:.6f})".format(model, sampling, roc_auc))
################################################################################################

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic -- {}'.format(dataset))
plt.legend(loc="lower right")
plt.savefig('{}_roc.pdf'.format(dataset), format='pdf')
plt.show()
