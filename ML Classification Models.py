#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Roger H Hayden III
#Udemy - The Complete Machine Learning Course with Python
#Classification Modeling techniques
#4/19/22


# In[137]:


#General Import
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn

#Logistic Regression
from sklearn.linear_model import LogisticRegression

#MNIST Dataset download
from sklearn.datasets import fetch_openml

#Train/Test Split
from sklearn.model_selection import train_test_split

#SGDClassifier
from sklearn.linear_model import SGDClassifier

#Stratified k-Fold
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

#Cross Validation
from sklearn.model_selection import cross_val_score

#Confusion Matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

#Precision
from sklearn.metrics import precision_score, recall_score

#F1 Score
from sklearn.metrics import f1_score

#Precision / Recall Curve
from sklearn.metrics import precision_recall_curve

#ROC Curve
from sklearn.metrics import roc_curve

#AUC Curve
from sklearn.metrics import roc_auc_score

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier


# In[3]:


x = np.linspace(-6, 6, num = 1000)
plt.figure(figsize = (12,8))
plt.plot(x, 1 / (1 + np.exp(-x))); # Sigmoid Function
plt.title("Sigmoid Function");


# In[4]:


#Example of providing a classification or a True, False output
tmp = [0, 0.4, 0.6, 0.8, 1.0]
tmp


# In[5]:


np.round(tmp)


# In[6]:


np.array(tmp) > 0.7


# In[7]:


#Producing yhat and rounded yhat from a provided data set and coefficients
dataset = [[-2.0011, 0],
           [-1.4654, 0],
           [0.0965, 0],
           [1.3881, 0],
           [3.0641, 0],
           [7.6275, 1],
           [5.3324, 1],
           [6.9225, 1],
           [8.6754, 1],
           [7.6737, 1]]
coef = [-0.806605464, 0.2573316]


# In[8]:


for row in dataset:
    yhat = 1.0 / (1.0 + np.exp(- coef[0] - coef[1] * row[0]))
    print("yhat {0:.4f}, yhat {1}".format(yhat, round(yhat)))


# =========================================================================

# Learning the Logistic Regression Model
# - Typically done with the maximization likelihood estimation
#     - Common learning algorithm
#     - search procedure that seeks the values for the coefficients that will minimize the error in the probabilities predicted by the model

# Learning with stochastic Gradient Descent
# - Logistic Regression uses gradient descent to update the coefficients

# ======================================================================

# Using Scikit Learn to Estimate Coefficients

# In[10]:


#Done Using Logisitic Regression


# In[11]:


dataset


# In[12]:


X = np.array(dataset)[:, 0:1]
y = np.array(dataset)[:, 1]


# In[13]:


clf_LR = LogisticRegression(C=1.0, penalty='l2', tol=0.0001, solver="lbfgs")


# In[14]:


clf_LR.fit(X,y)


# In[15]:


clf_LR.predict(X)


# In[16]:


#First column is probability of 0, second is probability of 1
clf_LR.predict_proba(X)


# =======================================================================

# Classification Exercise

# In[17]:


dataset2 = [[ 0.2,  0. ],
            [ 0.2,  0. ],
            [ 0.2,  0. ],
            [ 0.2,  0. ],
            [ 0.2,  0. ],
            [ 0.4,  0. ],
            [ 0.3,  0. ],
            [ 0.2,  0. ],
            [ 0.2,  0. ],
            [ 0.1,  0. ],
            [ 1.4,  1. ],
            [ 1.5,  1. ],
            [ 1.5,  1. ],
            [ 1.3,  1. ],
            [ 1.5,  1. ],
            [ 1.3,  1. ],
            [ 1.6,  1. ],
            [ 1. ,  1. ],
            [ 1.3,  1. ],
            [ 1.4,  1. ]]


# In[18]:


X = np.array(dataset2)[:, 0:1]
y = np.array(dataset2)[:, 1]


# In[19]:


clf_LR = LogisticRegression(C=1.0, penalty='l2', tol=0.0001, solver='lbfgs')

clf_LR.fit(X,y)


# In[20]:


y_pred = clf_LR.predict(X)
clf_LR.predict(X)


# In[21]:


np.column_stack((y_pred, y))


# In[22]:


clf_LR.predict(np.array([0.9]).reshape(1,-1))


# ======================================================================

# Downloading MNIST Dataset

# In[24]:


#Digit Recognition Dataset
mnist = fetch_openml(name='mnist_784')
mnist


# In[38]:


len(mnist['data'])


# Visualization

# In[39]:


X, y = mnist['data'], mnist['target']


# In[83]:


X.to_numpy()


# In[84]:


X.shape


# In[85]:


y.to_numpy()


# In[86]:


y = y.astype("float")


# In[87]:


X[69999]


# In[88]:


y[69999]


# In[89]:


y.shape


# In[90]:


def viz(n):
    plt.imshow(X[n].reshape(28,28))
    return


# In[36]:


viz(69999)


# In[49]:


y[1000]


# In[50]:


viz(1000)


# Locating Number 4 and plotting the image

# In[53]:


type(y)


# In[54]:


y == 4


# In[60]:


np.where(y==4)


# In[61]:


y[69977]


# In[62]:


_ = X[69977]
_image = _.reshape(28, 28)
plt.imshow(_image);


# In[63]:


viz(69977)


# Train/Test Split

# In[69]:


#Method 1
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)


# In[66]:


#Method 2
num_split = 60000
X_train, X_test, y_train, y_test = X[:num_split], X[num_split:], y[:num_split], y[num_split:]


# Shuffle Training Dataset

# In[68]:


#shuffle_index = np.random.permutation(num_split)
#X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# Training a Binary Classifier

# In[71]:


y_train_0 = (y_train == 0)
y_train_0


# In[72]:


y_test_0 = (y_test == 0)
y_test_0 


# SGDClassifier

# In[74]:


clf = SGDClassifier(random_state = 0)
clf.fit(X_train, y_train_0)

Prediction
# In[75]:


viz(1000)


# In[76]:


clf.predict(X[1000].reshape(1, -1))


# In[77]:


viz(1001)


# In[78]:


clf.predict(X[1001].reshape(1, -1))


# =======================================================================

# Performance Measures

# Stratified k-Fold

# In[81]:


clf = SGDClassifier(random_state=0)


# In[82]:


skfolds = StratifiedKFold(n_splits=3, random_state=100)


# In[91]:


for train_index, test_index in skfolds.split(X_train, y_train_0):
    clone_clf = clone(clf)
    X_train_fold = X_train[train_index]
    y_train_folds = (y_train_0[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_0[test_index])
    
    clone_clf.fit(X_train_fold, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print("{0:.4f}".format(n_correct / len(y_pred)))


# Cross Validation Score

# In[92]:


cross_val_score(clf, X_train, y_train_0, cv=3, scoring='accuracy')


# Confusion Matrix

# In[95]:


y_train_pred = cross_val_predict(clf, X_train, y_train_0, cv=3)


# In[96]:


confusion_matrix(y_train_0, y_train_pred)


# In[97]:


pd.DataFrame(confusion_matrix(y_train_0, y_train_pred))


# In[98]:


pd.DataFrame(confusion_matrix(y_train_0, y_train_pred),
             columns=pd.MultiIndex.from_product([['Prediction'], ["Negative", "Positive"]]),
             index=pd.MultiIndex.from_product([["Actual"], ["Negative", "Positive"]]))


# Precision

# In[100]:


precision_score(y_train_0, y_train_pred) # 5618 / (574 + 5618)


# Recall

# In[101]:


recall_score(y_train_0, y_train_pred) # 5618 / (305 + 5618)


# F1 Score

# In[103]:


f1_score(y_train_0, y_train_pred)


# Precision / Recall Tradeoff

# In[104]:


np.random.seed(0)
clf = SGDClassifier(random_state=0)
clf.fit(X_train, y_train_0)


# In[105]:


y_scores = clf.decision_function(X[1000].reshape(1, -1))
y_scores


# In[106]:


threshold = 0


# In[108]:


y_some_digits_pred = (y_scores > threshold)
y_some_digits_pred


# In[109]:


threshold = 40000
y_some_digits_pred = (y_scores > threshold)
y_some_digits_pred


# In[110]:


y_scores = cross_val_predict(clf, X_train, y_train_0, cv=3, method='decision_function')


# In[112]:


plt.figure(figsize=(12,8)); plt.hist(y_scores, bins=100);


# In[114]:


precisions, recalls, thresholds = precision_recall_curve(y_train_0, y_scores)


# In[116]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([-0.5,1.5])  


# In[117]:


plt.figure(figsize=(12,8)); 
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# In[118]:


plt.figure(figsize=(12,8)); 
plt.plot(precisions, recalls);
plt.xlabel('recalls');
plt.ylabel('precisions');
plt.title('PR Curve: precisions/recalls tradeoff');


# High Precisions (> 0.90)

# In[119]:


plt.figure(figsize=(12,8)); 
plt.plot(thresholds, precisions[1:]);


# In[120]:


idx = len(precisions[precisions < 0.9])


# In[121]:


thresholds[idx]


# In[122]:


y_train_pred_90 = (y_scores > 21454)


# In[123]:


recall_score(y_train_0, y_train_pred_90)


# High Precisions (> 0.99)

# In[126]:


idx = len(precisions[precisions < 0.99])


# In[127]:


y_train_pred_99 = (y_scores > thresholds[idx])


# In[128]:


precision_score(y_train_0, y_train_pred_99)


# In[129]:


recall_score(y_train_0, y_train_pred_99)


# ROC Curve

# In[131]:


fpr, tpr, thresholds = roc_curve(y_train_0, y_scores)


# In[132]:


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')


# In[133]:


plt.figure(figsize=(12,8)); 
plot_roc_curve(fpr, tpr)
plt.show();


# AUC 

# In[136]:


roc_auc_score(y_train_0, y_scores)


# In[138]:


f_clf = RandomForestClassifier(random_state=0, n_estimators=100)


# In[139]:


y_probas_forest = cross_val_predict(f_clf, X_train, y_train_0,
                                    cv=3, method='predict_proba')


# In[140]:


y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_0, y_scores_forest)


# In[141]:


plt.figure(figsize=(12,8)); 
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show();


# In[142]:


roc_auc_score(y_train_0, y_scores_forest)


# In[143]:


f_clf.fit(X_train, y_train_0)


# In[144]:


y_train_rf = cross_val_predict(f_clf, X_train, y_train_0, cv=3)


# In[145]:


precision_score(y_train_0, y_train_rf) 


# In[146]:


recall_score(y_train_0, y_train_rf) 


# In[147]:


confusion_matrix(y_train_0, y_train_rf)

