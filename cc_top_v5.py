# -*- coding: utf-8 -*-
"""
Created on Thu April 28 20:46:45 2017

@author: farismismar
"""
import theano  # # Must use Theano, or else the whole thing will fail
from theano.tensor.shared_randomstreams import RandomStreams

from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
from sklearn.model_selection import train_test_split

from matplotlib import rc
import numpy as np

import os
os.chdir('/Users/farismismar/Desktop/Credit Card/xtr0')

##### TO DO:
################################
# Needed for the dataset to generate proper latex table
data_mnemonic = 'CC'

# Set the random seed
seed = 123
np.random.seed(seed)

srng = RandomStreams(seed)

# Import the model to memory first
dataset = pd.read_csv('../Datasets/creditcard.csv')

# Unwanted columns
#dataset.drop(['AppointmentRegistration', 'ApointmentData'], axis=1, inplace=True)

##### TO DO:
################################
# change train['xxxxx'] and test['xxxxx'] to the proper class column

import matplotlib.pyplot as plt

# Sanity check. Missing values?
print('Number of missing values: {}'.format(dataset.isnull().sum().sum()))

data_summary = dataset['Class'].value_counts()
n = data_summary.shape[0]

# Is the problem imbalanced?
print(data_summary)

plt.figure(figsize=(15,3))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(range(n), data_summary.values)
plt.title('Target variable -- before')
plt.xticks(range(n), data_summary.index)
plt.savefig('hr_balance.pdf', format='pdf')

# Perform a split 30-70
train, test = train_test_split(dataset, test_size = 0.30, random_state = seed)

# TO DO
# Ordinal variables
# No need for dummy coding here, just a quick replace.
#train['Status'].replace(['Show-Up', 'No-Show'], [0, 1],inplace=True)
#test['Status'].replace(['Show-Up', 'No-Show'], [0, 1],inplace=True)

X_train = train.drop('Class', axis = 1)
X_test = test.drop('Class', axis = 1)

y_train = train['Class'].values
y_test = test['Class'].values


# TO DO
###### Categorical variables
# Do you need dummy encoding?


dummy_encoding = True

if dummy_encoding == True:
    # Dummy coding is required here
    X_train = pd.get_dummies(X_train).values
    X_test = pd.get_dummies(X_test).values


nX, mX = X_train.shape

# The minority class is 1
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Train 0: original imbalanced
xtr0 = X_train
ytr0 = y_train

# Train 1: use SMOTE and K = 5 (default)
sm = SMOTE(random_state=seed)
xtr1, ytr1 = sm.fit_sample(X_train, y_train)

#
# TO DO: replace all train[''] and test[''] with the proper class column name
try:
    # Train 2: 1% Undersampling
    train_c1 = train[(y_train == 1)]
    train_c0 = train[(y_train == 0)].sample(train_c1.shape[0]*99, random_state=seed)
    train2 = train_c1.append(train_c0)
    xtr2 = train2.drop('Class', axis = 1)
    ytr2 = train2['Class']
except:
    print('Undersampling 1% failed')
    pass

try:
    # Train 3: 5% Undersampling
    train_c1 = train[(y_train == 1)]
    train_c0 = train[(y_train == 0)].sample(train_c1.shape[0]*19, random_state=seed)
    train3 = train_c1.append(train_c0)
    xtr3 = train3.drop('Class', axis = 1)
    ytr3 = train3['Class']
except:
    print('Undersampling 5% failed')
    pass

try:
    # Train 4: 10% Undersampling
    train_c1 = train[(y_train  == 1)]
    train_c0 = train[(y_train == 0)].sample(train_c1.shape[0]*9, random_state=seed)
    train4 = train_c1.append(train_c0)
    xtr4 = train4.drop('Class', axis = 1)
    ytr4 = train4['Class']
except:
    print('Undersampling 10% failed')
    pass

try:
    # Train 5: 20% Undersampling
    train_c1 = train[(y_train  == 1)]
    train_c0 = train[(y_train == 0)].sample(train_c1.shape[0]*4, random_state=seed)
    train5 = train_c1.append(train_c0)
    xtr5 = train5.drop('Class', axis = 1)
    ytr5 = train5['Class']
except:
    print('Undersampling 20% failed')
    pass

try:
    # Train 6: 50% Undersampling
    train_c1 = train[(y_train  == 1)]
    train_c0 = train[(y_train == 0)].sample(train_c1.shape[0], random_state=seed)
    train6 = train_c1.append(train_c0)
    xtr6 = train6.drop('Class', axis = 1)
    ytr6 = train6['Class']
except:
    pass

##### TO DO:
################################
# Replace xtr0, ytr0 with xtr1, ytr1 and collect the outputs

try:
    X_train, y_train = xtr0, ytr0
except:
    f = open('Config_Invalid.txt', 'w')
    f.close()
    quit()
try:
    if dummy_encoding == True:
        X_train = pd.get_dummies(X_train).values
except:
    pass

data_summary = pd.Series(y_train).value_counts()
n = data_summary.shape[0]

print(data_summary)

plt.figure(figsize=(15,3))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.bar(range(n), data_summary.values)
plt.xticks(range(n), data_summary.index)
plt.title('Distribution of data')
plt.savefig('balance_after.pdf', format='pdf')

try:
    mY, nY = y_train.shape
except:
    mY = y_train.shape[0]
    nY = 1

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

neigh = KNeighborsClassifier(n_jobs=-1)

hyperparameters = {'n_neighbors': np.arange(19) + 1}

gs_neigh = GridSearchCV(neigh, hyperparameters, scoring='roc_auc', cv=3)
gs_neigh.fit(X_train, y_train)

# This is the best model
best_model_knn = gs_neigh.best_params_
print(best_model_knn)

neigh = gs_neigh.best_estimator_
neigh.fit(X_train, y_train)

y_hat_knn = neigh.predict(X_test)
y_score_knn = neigh.predict_proba(X_test)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute ROC curve and ROC area
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_score_knn[:,1])
roc_auc_knn = auc(fpr_knn, tpr_knn)

plt.figure(figsize=(13,8))
lw = 2

plt.plot(fpr_knn, tpr_knn,
     lw=lw, label="ROC curve (AUC = {:.6f})".format(roc_auc_knn))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic -- KNN')
plt.legend(loc="lower right")
plt.savefig('knn_roc.pdf', format='pdf')
#plt.show()

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_hat_gnb = gnb.predict(X_test)
y_score_gnb = gnb.predict_proba(X_test)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute ROC curve and ROC area
fpr_gnb, tpr_gnb, _ = roc_curve(y_test, y_score_gnb[:,1])
roc_auc_gnb = auc(fpr_gnb, tpr_gnb)

plt.figure(figsize=(13,8))
lw = 2

plt.plot(fpr_gnb, tpr_gnb,
     lw=lw, label="ROC curve (AUC = {:.6f})".format(roc_auc_gnb))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic -- Naive Bayes')
plt.legend(loc="lower right")
plt.savefig('gnb_roc.pdf', format='pdf')
#plt.show()

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K

mX, nX = X_train.shape
mY = y_train.shape[0]
nY = 1

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train_sc = ss.fit_transform(X_train)
X_test_sc = ss.transform(X_test)

# create model
def create_mlp(optimizer, output_dim, n_hidden):
    mlp = Sequential()
    mlp.add(Dense(units=output_dim, input_dim=nX, activation='sigmoid'))
    for k in np.arange(n_hidden):
        mlp.add(Dense(output_dim, use_bias=True))

    mlp.add(Dense(units=nY, input_dim=output_dim, activation='sigmoid', use_bias=True))

    mlp.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', log_loss])  # log loss is an alternative for AUC; ignore accuracy
    return mlp

def log_loss(y_true, y_pred):
   return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

model = KerasClassifier(build_fn=create_mlp, verbose=1, epochs=5, batch_size=16)

# The hyperparameters
optimizers = ['sgd', 'adam']
output_dims=[3,5]
n_hiddens = [1,3]

hyperparameters = dict(optimizer=optimizers, output_dim=output_dims, n_hidden=n_hiddens)

grid = GridSearchCV(estimator=model, param_grid=hyperparameters, n_jobs=1, cv=3)
grid_result = grid.fit(X_train_sc, y_train)

# This is the best model
best_model_mlp = grid_result.best_params_
print(best_model_mlp)

# The best results are
# {'n_hidden': 1, 'optimizer': 'sgd', 'output_dim': 1}

mlp = Sequential()
mlp.add(Dense(units=grid_result.best_params_['output_dim'], input_dim=nX, activation='sigmoid'))

for k in np.arange(grid_result.best_params_['n_hidden']):
    mlp.add(Dense(grid_result.best_params_['output_dim'], use_bias=True)) # no sigmoid here

mlp.add(Dense(units=nY, input_dim=grid_result.best_params_['output_dim'], activation='sigmoid', use_bias=True))

# Compile model with accuracy metric
mlp.compile(loss='binary_crossentropy', optimizer=grid_result.best_params_['optimizer'], metrics=['accuracy', log_loss])
mlp.fit(X_train_sc, y_train, epochs=32, batch_size=16)

# Create mlp object with params from grid_result then generate these

y_score_mlp = mlp.predict_proba(X_test_sc)
y_hat_mlp = mlp.predict_classes(X_test_sc)


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute ROC curve and ROC area
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_score_mlp)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

plt.figure(figsize=(13,8))
lw = 2

plt.plot(fpr_mlp, tpr_mlp,
     lw=lw, label="ROC curve (AUC = {:.6f})".format(roc_auc_mlp))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic -- MLP')
plt.legend(loc="lower right")
plt.savefig('mlp_roc.pdf', format='pdf')
#plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

mX, nX = X_test.shape
n0 = y_train[y_train==0].shape[0]
n1 = y_train[y_train==1].shape[0]
class_weight = {0:n0/(n0+n1), 1:n1/(n0+n1)}

clf = RandomForestClassifier(class_weight=class_weight,
                                bootstrap = True,
                                n_estimators = 100,
                                oob_score = True, random_state=seed)

hyperparameters = {'max_depth': [3, None],
                  'max_features': [min(10, nX), None],
                  'criterion': ['entropy', 'gini'],
                  'min_samples_split': [2, 10]}

gs_clf = GridSearchCV(clf, hyperparameters, scoring='roc_auc', cv=3)
gs_clf.fit(X_train, y_train)


# This is the best model
best_model_rf = gs_clf.best_params_
print(best_model_rf)

clf = gs_clf.best_estimator_

clf.fit(X_train, y_train)
y_score_rf = clf.predict_proba(X_test)
y_hat_rf = clf.predict(X_test)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute ROC curve and ROC area
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf[:,1])
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(13,8))
lw = 2

plt.plot(fpr_rf, tpr_rf,
     lw=lw, label="ROC curve (AUC = {:.6f})".format(roc_auc_rf))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - RandomForest')
plt.legend(loc="lower right")
plt.savefig('rf_roc.pdf', format='pdf')
#plt.show()

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

classifier = xgb.XGBClassifier(seed=seed, silent=False, colsample_bytree=0.7,
                             learning_rate=0.05, n_estimators = 100)

#classifier.get_params().keys()

# Hyperparameters
alphas = np.linspace(0,1,4)
lambdas = np.linspace(0,1,4)
depths=[2,8]
objectives = ['binary:logistic', 'reg:linear']

hyperparameters = {'reg_alpha': alphas, 'reg_lambda': lambdas, 'objective': objectives, 'max_depth': depths}

gs_xgb = GridSearchCV(classifier, hyperparameters, scoring='roc_auc', cv=3) # 5-fold crossvalidation
gs_xgb.fit(X_train, y_train)

# This is the best model
best_model_xgb = gs_xgb.best_params_
print(best_model_xgb)

clf = gs_xgb.best_estimator_

clf.fit(X_train, y_train, eval_metric='auc')

y_score_xgb = clf.predict_proba(X_test)
y_hat_xgb = clf.predict(X_test)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute ROC curve and ROC area
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_score_xgb[:,1])
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

plt.figure(figsize=(13,8))
lw = 2

plt.plot(fpr_xgb, tpr_xgb,
     lw=lw, label="ROC curve (AUC = {:.6f})".format(roc_auc_xgb))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic -- XGBoost')
plt.legend(loc="lower right")
plt.savefig('xgboost_roc.pdf', format='pdf')
#plt.show()

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import KFold

kFold = 3
cv = KFold(n_splits=kFold, random_state=seed)

# Default is accuracy_score.
clf = LogisticRegressionCV(penalty='l2', cv=cv, random_state=seed)

clf.fit(X_train, y_train) # for the ith class
C_optimal = clf.C_[0]

# This is the best model
best_model_lr = clf.C_[0]
print(best_model_lr)

clf = LogisticRegression(penalty='l2', random_state=seed, C=C_optimal)
clf.fit(X_train, y_train)
y_score_logistic = clf.predict_proba(X_test)
y_hat_logistic = clf.predict(X_test)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute ROC curve and ROC area
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_logistic[:,1])
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize=(13,8))
lw = 2

plt.plot(fpr_lr, tpr_lr,
     lw=lw, label="ROC curve (AUC = {:.6f})".format(roc_auc_lr))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic -- Logistic Regression')
plt.legend(loc="lower right")
plt.savefig('logit_roc.pdf', format='pdf')
#plt.show()

plt.figure(figsize=(13,8))
lw = 2

for (fpr, tpr, auc, model) in zip([fpr_knn, fpr_gnb, fpr_lr, fpr_mlp, fpr_xgb],
                                  [tpr_knn, tpr_gnb, tpr_lr, tpr_mlp, tpr_xgb],
                                  [roc_auc_knn, roc_auc_gnb, roc_auc_lr, roc_auc_mlp, roc_auc_lr, roc_auc_xgb],
                                  ['KNN', 'NB', 'LR', 'MLP', 'XGB']):
    plt.plot(fpr, tpr,
         lw=lw, label="ROC curve {} (AUC = {:.6f})".format(model, auc))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC.pdf', format='pdf')
#plt.show()

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
import itertools

# http://www.kdnuggets.com/2016/03/lift-analysis-data-scientist-secret-weapon.html
def lift_score(y_test, y_hat):
    numerator = accuracy_score(y_test[y_test==1], y_hat[y_test==1])
    denominator = y_test[y_test==1].shape[0] / y_test.shape[0]
    return numerator / denominator

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

target = open('metrics.txt', 'w')
target.write('\\multirow{6}{*}{%s}\n' % data_mnemonic)

for (y_hat, model) in zip([y_hat_knn, y_hat_gnb, y_hat_logistic, y_hat_mlp, y_hat_xgb, y_hat_rf],
                            ['KNN', 'NB', 'LR', 'MLP', 'XGB', 'RF']):
    p, r, fsc, _ = precision_recall_fscore_support(y_test, y_hat)
    acc = accuracy_score(y_test, y_hat)
    c1_acc = accuracy_score(y_test[y_test==1], y_hat[y_test==1])
    c0_acc = accuracy_score(y_test[y_test==0], y_hat[y_test==0])

    # tpr is given by     tpr_gnb, tpr_knn, ....
    # roc is given by     roc_auc_gnb, roc_auc_knn, ....
    if (model=='KNN'):
        tpr, fpr, roc = tpr_knn, fpr_knn, roc_auc_knn
        best_model = best_model_knn
    if (model=='NB'):
        tpr, fpr, roc = tpr_gnb, fpr_gnb, roc_auc_gnb
        best_model = 'N/A'
    if (model=='LR'):
        tpr, fpr, roc = tpr_lr, fpr_lr, roc_auc_lr
        best_model = best_model_lr
    if (model=='MLP'):
        tpr, fpr, roc = tpr_mlp, fpr_mlp, roc_auc_mlp
        best_model = best_model_mlp
    if (model=='XGB'):
        tpr, fpr, roc = tpr_xgb, fpr_xgb, roc_auc_xgb
        best_model = best_model_xgb
    if (model=='RF'):
        tpr, fpr, roc = tpr_rf, fpr_rf, roc_auc_rf
        best_model = best_model_rf

    lift = lift_score(y_test, y_hat)
    logloss = log_loss(y_test, y_hat)

    target.write('& {0} & {1:.4f} & {2:.4f} & {3:.4f} & {4:.4f} & {5:.4f} & {6:.4f} & {7:.4f} & {8:.4f} & {9:.4f} \\\\'.format(
        model, lift, acc, c1_acc, c0_acc, fsc[1], p[1], r[1], roc, logloss))
    target.write('\n')
    target.write('\\hline\n')
    target.write('---------------\n')
    target.write('Optimal params: {}\n'.format(best_model))
    target.write('TPR vector: \n')
    target.write(''.join(["{}, ".format(str(i)) for i in tpr]) + '\n')
    target.write('FPR vector: \n')
    target.write(''.join(["{}, ".format(str(i)) for i in fpr]) + '\n')
    target.write('==================================\n')

    cm = confusion_matrix(y_test, y_hat)

    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cm, classes=[0,1], normalize=True,
                          title='Normalized confusion matrix -- {}'.format(model))
    plt.savefig('confusion_matrix_{}.pdf'.format(model), format='pdf')
    #plt.show()
target.close()
