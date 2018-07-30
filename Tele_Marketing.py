
# coding: utf-8

# 
# -------------
# 
# <h3 align='center'> WCD Machine Learning Project </h3>
# <h1 align='center'> Predicting the success of bank telemarketing in Portugal </h1>
# 
# 
# <br>
# <center align="left"> Developed by: </center>
# <center align="left"> Babak Khamsehi </center>
# 
# 

# <img src='Workflow.png'>

# ### Import some libraries and reading in the data file

# In[41]:


import pandas as pd
bank= pd.read_csv('C:/Babak/Data Science/Machine Learning/project/banking/bank.csv',sep=';')
X=bank.drop(['duration', 'y'],axis=1)   # Dropping duration (irrelevant for prediction) and the y
Y=bank['y']                             # y=has the client subscribed (yes/no)
Y=pd.DataFrame(Y)
bank.head()


# ### Exploring the features

# In[42]:


# Slicing bank data into numerical and categorical dataframes 
X_num=bank[['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed' ]]
X_cat=X.drop(X_num, axis=1)
X_num.head()


# In[43]:


X_cat.head()


# ### Frequency tables for categorical features

# In[44]:


# convert non-numeric to factors
# job and job cat
X["job_cat"] = pd.factorize(X["job"])[0]
# cross tables 
freq_job = pd.crosstab(index=[X["job_cat"], X["job"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_job) 

# martial and marital cat
X["marital_cat"] = pd.factorize(X["marital"])[0]
# cross tables 
freq_marital = pd.crosstab(index=[X["marital_cat"], X["marital"]],  # Make a crosstab
                        columns="count", margins= True)
print(freq_marital) 

freq_marital = pd.crosstab(index=[X["marital_cat"], X["marital"]],  # Make a crosstab
                        columns="count", margins= True)

# education and education cat
X["education_cat"] = pd.factorize(X["education"])[0]
# cross tables 
freq_education = pd.crosstab(index=[X["education_cat"], X["education"]],  # Make a crosstab
                        columns="count", margins= True)
print(freq_education) 

# default and default cat: Has credit ? 
X["default_cat"] = pd.factorize(X["default"])[0]
# cross tables 
freq_default = pd.crosstab(index=[X["default_cat"], X["default"]],  # Make a crosstab
                        columns="count", margins= True)
print(freq_default) 

# housing and housing cat
X["housing_cat"] = pd.factorize(X["housing"])[0]
# cross tables 
freq_housing = pd.crosstab(index=[X["housing_cat"], X["housing"]],  # Make a crosstab
                        columns="count", margins= True)
print(freq_housing) 

# loan and loan cat : has personal loan? 
X["loan_cat"] = pd.factorize(X["loan"])[0]
# cross tables 
freq_loan = pd.crosstab(index=[X["loan_cat"], X["loan"]],  # Make a crosstab
                        columns="count", margins= True)
print(freq_loan)
print()
# contact and contact cat : contact rout
X["contact_cat"] = pd.factorize(X["contact"])[0]
# cross tables 
freq_contact = pd.crosstab(index=[X["contact_cat"], X["contact"]],  # Make a crosstab
                        columns="count", margins= True)
print(freq_contact)

# month and month cat 
X["month_cat"] = pd.factorize(X["month"])[0]
# cross tables 
freq_month = pd.crosstab(index=[X["month_cat"], X["month"]],  # Make a crosstab
                        columns="count", margins= True)
print(freq_month)

# day_of_week and day_of_week cat 
X["day_of_week_cat"] = pd.factorize(X["day_of_week"])[0]
# cross tables 
freq_day_of_week = pd.crosstab(index=[X["day_of_week_cat"], X["day_of_week"]],  # Make a crosstab
                        columns="count", margins= True)
print(freq_day_of_week)

# poutcome: previous outcome (of telemarketting)
X["poutcome_cat"] = pd.factorize(X["poutcome"])[0]
# cross tables 
freq_poutcome = pd.crosstab(index=[X["poutcome_cat"], X["poutcome"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_poutcome)
X.head(10)


#  ### Dummy coding for 10 categorical variables
#       The first cateogorical feature is the reference group
#       Original  categorical varibales will be deleted 

# In[45]:


# Dummy coding for 10 categorical variables
X1= pd.get_dummies(X, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome'], drop_first=True) #


Y_d= pd.get_dummies(Y.y, prefix='y').iloc[:,1:]  # y=has the client subscribed (yes/no)
Y= pd.concat([Y, Y_d], axis=1)

import numpy as np
np.bincount(Y.y_yes)    # Negative class (0) is the most frequent class


# ## Splitting into training and test datasets

# In[76]:


# IMP: Dropping columns with names 
X2= X1[X1.columns.drop(list(X1.filter(regex='_cat')))]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, Y.y_yes, random_state=0)


# ## Feature importance using Decision Tree Classifier

# In[77]:


#from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree

clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))

print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# In[78]:


from adspy_shared_utilities import plot_feature_importances
import numpy as np, matplotlib.pyplot as plt

plt.figure(figsize=(10,40), dpi=180)
plot_feature_importances(clf, X_train.columns)
plt.show()

#print('Feature importances: {}'.format(clf.feature_importances_))


# In[79]:


df = pd.DataFrame({'feature names':X_train.columns, 'feature importance':clf.feature_importances_})
df= df.sort_values(['feature importance'], ascending=False)
df


# ## Training datasets with only two most important features 

# In[80]:


X2_train = X_train.loc[:,['age','nr.employed']] # only 2 features 
X2_test = X_test.loc[:,['age','nr.employed']]


# In[81]:


X2_train.head()


# In[82]:


X2_test.head()


# In[83]:


from matplotlib.colors import ListedColormap
cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])
                          
import matplotlib.pyplot as plt
plt.figure()
plt.title('This is a Binary classification problem with non-linearly separable classes')
plt.scatter(X2_train.iloc[:,0], X2_train.iloc[:,1], c=y_train,
           marker= 'o', s=50, cmap=cmap_bold)
plt.show()


# ## Comparisons of diffierent classifiers
#     1. Establishing a baseline with dummy classifiers (majority vote and stratified)
#     2. Presenting real classifiers 

# ### Majority-vote-based dummy classifiers 

# In[84]:


np.bincount(y_train) 
from sklearn.dummy import DummyClassifier
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train) # Negative class (0) is most frequent

dummy_majority.score(X_test, y_test) # Mean accuracy score 
y_dum_major_pred = dummy_majority.predict(X_test)  # Predicts every y as 0 


# Combined report with all above metrics
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score

# Classification report
print(classification_report(y_test, y_dum_major_pred, target_names=['not 1', '1']))
# fbeta score, beta =0.50, more weight for precision rather than recall
fbeta_score(y_test, y_dum_major_pred, beta=0.50)


# ### Stratified-based dummy classifiers 
# 

# In[85]:


# produces random predictions w/ same class proportion as training set
dummy_strat = DummyClassifier(strategy='stratified').fit(X_train, y_train)
y_dum_major_strat_pred = dummy_strat.predict(X_test)
np.bincount(y_dum_major_strat_pred) # exactly the same as bincounts for training dataset

# Classification report
print(classification_report(y_test, y_dum_major_strat_pred, target_names=['not 1', '1']))
# fbeta score, beta =0.50, more weight for precision rather than recall
fbeta_score(y_test, y_dum_major_strat_pred, beta=0.50)


# In[97]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


#dataset = load_digits()
#X, y = dataset.data, dataset.target == 1
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Create a two-feature input vector matching the example plot above
# We jitter the points (add a small amount of random noise) in case there are areas
# in feature space where many instances have the same features.
# jitter_delta = 0.25
X_2_train = X_train.loc[:,['age','nr.employed']]
X_2_test = X_test.loc[:,['age','nr.employed']] 

X_2_test


# # Real classifiers

# ## Linear Logistic Regression 

# In[86]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

lr = LogisticRegression().fit(X_train, y_train)
lr_predicted = lr.predict(X_test)
confusion = confusion_matrix(y_test, lr_predicted)

print('Logistic regression classifier (default settings)\n', confusion)

# Classification report
print(classification_report(y_test, lr_predicted, target_names=['not 1', '1']))
# fbeta score, beta =0.50, more weight for precision rather than recall
fbeta_score(y_test, lr_predicted, beta=0.50)


# ## Support Vector Machines Classifer 

# ## Gussian SVM

# In[ ]:


from sklearn.svm import SVC
# Gussian, with C=1, and gamma=0.2 
svm= SVC(C=1, kernel='rbf', gamma=0.2).fit(X_train, y_train)
# svm.score(X_test, y_test)
svm_pred_g = svm.predict(X_test)  # Gussian, C=1

confusion = confusion_matrix(y_test, lr_predicted)
print('Support Vector Machine classifier (C=1, Gussian, gamma=0.2)\n', confusion)

# Classification report
print(classification_report(y_test, svm_pred_g, target_names=['not 1', '1']))
# fbeta score, beta =0.50, more weight for precision rather than recall

fbeta_score(y_test, svm_pred_g, beta=0.50)


# In[ ]:


## Linear SVM


# In[ ]:


svm= SVC(C=1, kernel='linear', gamma=0.2).fit(X_train, y_train)
# svm.score(X_test, y_test)
svm_pred_l = svm.predict(X_test)  # Gussian, C=1

# Confusion matrix
confusion = confusion_matrix(y_test, svm_pred_l)
print('Support Vector Machine (C=1, Linear, gamma=0.2)\n', confusion)

# Classification report
print(classification_report(y_test, svm_pred_l, target_names=['not 1', '1']))
# fbeta score, beta =0.50, more weight for precision rather than recall
fbeta_score(y_test, svm_pred_l, beta=0.50)


# ## Decission Tree Classifier

# ### Decision Tree, Max Depth=2
# 

# In[ ]:


# Decision Tree, Max Depth=2
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
tree_predicted = dt.predict(X_test)
# Confusion matrix
confusion = confusion_matrix(y_test, tree_predicted)
print('Decision tree classifier (max_depth = 2)\n', confusion)
# Classification report
print(classification_report(y_test, tree_predicted, target_names=['not 1', '1']))
# fbeta score, beta =0.50, more weight for precision rather than recall
fbeta_score(y_test, tree_predicted, beta=0.50)


# # Models Evaluations

# ## Precision-recall curves based on 
#     Decision function scores
#     Probablity scores 

# In[88]:


X_train, X_test, y_train, y_test = train_test_split(X1, Y.y_yes, random_state=0)
y_scores_lr = lr.fit(X_train, y_train).decision_function(X_test)
y_score_list = list(zip(y_test[0:20], y_scores_lr[0:20]))

# show the decision_function scores for first 20 instances
y_score_list

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)
closest_zero = np.argmin(np.abs(thresholds))
closest_zero_p = precision[closest_zero]
closest_zero_r = recall[closest_zero]

plt.figure()
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.plot(precision, recall, label='Precision-Recall Curve')
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
plt.xlabel('Precision', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.axes().set_aspect('equal')
plt.show()


# In[89]:


X_train, X_test, y_train, y_test = train_test_split(X1, Y.y_yes, random_state=0)
y_proba_lr = lr.fit(X_train, y_train).predict_proba(X_test)
y_proba_list = list(zip(y_test[0:20], y_proba_lr[0:20,1]))

# show the probability of positive class for first 20 instances
y_proba_list


precision, recall, thresholds = precision_recall_curve(y_test, y_proba_lr[:,1])
#closest_zero = np.argmin(np.abs(thresholds))
#closest_zero_p = precision[closest_zero]
#closest_zero_r = recall[closest_zero]

plt.figure()
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.plot(precision, recall, label='Precision-Recall Curve')
#plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
plt.xlabel('Precision', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.axes().set_aspect('equal')
plt.show()


# ## ROC curves 
#     For Logistic Regression Classifier
#     For SVM Classifier 
#     For Decision Tree Classifier 

# In[90]:


## ROC curves Logistic regression Decision function scores versus probablit scores
from sklearn.metrics import roc_curve, auc

X_train, X_test, y_train, y_test = train_test_split(X1, Y.y_yes, random_state=0)

# Decision Function scores

y_score_lr = lr.fit(X_train, y_train).decision_function(X_test)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Probability scores
y_proba_lr = lr.fit(X_train, y_train).predict_proba(X_test)[:,1]
fpr_lr_p, tpr_lr_p, _ = roc_curve(y_test, y_proba_lr)
roc_auc_lr_p = auc(fpr_lr_p, tpr_lr_p)

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_lr, tpr_lr,'g.', lw=3, label='Descision Function (AUC = {:0.2f})'.format(roc_auc_lr))
plt.plot(fpr_lr_p, tpr_lr_p,'r:', lw=3, label='Probability (AUC = {:0.2f})'.format(roc_auc_lr_p))

plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve (Logistic Regression classifier)', fontsize=16)
plt.legend(loc='best', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, label= 'By Chance only', linestyle='--')
plt.legend(loc='best', fontsize=13)
plt.axes().set_aspect('equal')
plt.show()


# ## SVM, rbf (Gussian), C=1 , Testing different gammas's 
# 

# In[91]:


plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])

for g in [0.01, 0.1, 0.20, 1]: # different gammas
    
    svm = SVC(gamma=g).fit(X_train, y_train)
    y_score_svm = svm.decision_function(X_test)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    accuracy_svm = svm.score(X_test, y_test)
    print("gamma = {:.2f}  accuracy = {:.2f}   AUC = {:.2f}".format(g, accuracy_svm, 
                                                                    roc_auc_svm))
    plt.plot(fpr_svm, tpr_svm, lw=3, alpha=0.7, 
             label='SVM (gamma = {:0.2f}, AUC = {:0.2f})'.format(g, roc_auc_svm))

plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate (Recall)', fontsize=16)
plt.plot([0, 1], [0, 1], color='k', lw=0.5, linestyle='--')
plt.legend(loc="best", fontsize=11)
plt.title('ROC curve: SVM, Gussian Kernel, C=1 ', fontsize=16)
plt.axes().set_aspect('equal')

plt.show()


# ## SVM, rbf (Gussian), g=0.2 , Testing different C's 

# In[119]:


plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])

for C in [0.001, 0.01, 0.1, 1, 10 ]: # different C's
    g=0.2 
#for C in [1e-2, 0.1, 1e2]
    svm = SVC(C=C, gamma=g).fit(X_train, y_train)
    y_score_svm = svm.decision_function(X_test)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    accuracy_svm = svm.score(X_test, y_test)
    print("C = {:.3f}  accuracy = {:.3f}   AUC = {:.3f}".format(C, accuracy_svm, 
                                                                    roc_auc_svm))
    plt.plot(fpr_svm, tpr_svm, lw=3, alpha=0.7, 
             label='SVM (C = {:0.3f}, AUC = {:0.3f})'.format(C, roc_auc_svm))

plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate (Recall)', fontsize=16)
plt.plot([0, 1], [0, 1], color='k', lw=0.5, linestyle='--')
plt.legend(loc="best", fontsize=11)
plt.title('ROC curve: SVM(Gussian), g=0.2 ', fontsize=16)
plt.axes().set_aspect('equal')
plt.show()


# ## ROC curve for  Decision tree classifier

# In[122]:


plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])

for md in [2, 3, 4, 5]:
    dt = DecisionTreeClassifier(max_depth=md).fit(X_train, y_train)
    tree_predicted = dt.predict(X_test)
    
    fpr_tree, tpr_tree, _ = roc_curve(y_test, tree_predicted)
    roc_auc_tree = auc(fpr_tree, tpr_tree)

    accuracy_tree = dt.score(X_test, y_test)
    print("accuracy = {:.2f} , AUC = {:.3f}".format(accuracy_tree, roc_auc_tree))
    plt.plot(fpr_tree, tpr_tree, lw=3, alpha=0.7, 
             label='max depth = {:.3f} , (AUC = {:0.2f})'.format(md , roc_auc_tree))

plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate (Recall)', fontsize=16)
plt.plot([0, 1], [0, 1], color='k', lw=0.5, linestyle='--')
plt.legend(loc="best", fontsize=11)
plt.title('ROC curve: Decision tree with differe Max Depths ', fontsize=16)
plt.axes().set_aspect('equal')
plt.show()


# ## Recap of model evaluations

# <img src='compare.png'>

# <img src='F1_Fbeta.png' width=70%>

# # Model tuning

# In[121]:


## Model tuning for SVM, rbf
clf = SVC(kernel = 'rbf').fit(X2_train, y_train)
grid_values = {'class_weight':['balanced', {1:2},{1:3},{1:4},{1:5},{1:10},{1:20},{1:50}]}

for i, eval_metric in enumerate(('precision','recall', 'f1','roc_auc')):
    grid_clf_custom = GridSearchCV(clf, param_grid=grid_values, scoring=eval_metric)
    grid_clf_custom.fit(X2_train, y_train)
    print('Grid best parameter (max. {0}): {1}'
          .format(eval_metric, grid_clf_custom.best_params_))
    print('Grid best score ({0}): {1}'
          .format(eval_metric, grid_clf_custom.best_score_))


# <img src='goodbye.png' width=60%>
