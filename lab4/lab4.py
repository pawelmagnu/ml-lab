#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
from sklearn.model_selection import train_test_split


# In[2]:


import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# In[3]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
# print(data_breast_cancer['DESCR'])


# In[4]:


X,y = data_breast_cancer.data, data_breast_cancer.target
X = X.loc[:,['mean area','mean smoothness']]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)


# In[5]:


svm_ns = LinearSVC(C=1, loss = "hinge", random_state=42)
svm_clf = Pipeline([('scaler', StandardScaler()),
                    ('linear_svc', LinearSVC(C=1, loss='hinge', random_state=42))])


# In[6]:


svm_ns.fit(X_train,y_train)
svm_clf.fit(X_train,y_train)


# In[7]:


no_scale_acc_train = accuracy_score(y_train, svm_ns.predict(X_train))
no_scale_acc_test = accuracy_score(y_test, svm_ns.predict(X_test))

scale_acc_train = accuracy_score(y_train, svm_clf.predict(X_train))
scale_acc_test = accuracy_score(y_test, svm_clf.predict(X_test))


# In[8]:


bc_acc = [no_scale_acc_train, no_scale_acc_test, scale_acc_train, scale_acc_test]
bc_acc


# In[9]:


with open('bc_acc.pkl', 'wb') as f:
    pickle.dump(bc_acc,f)


# In[10]:


data_iris = datasets.load_iris(as_frame=True)
# print(data_iris['DESCR'])


# In[11]:


X_i, y_i = data_iris.data, data_iris.target
X_i = X_i.loc[:,['petal length (cm)','petal width (cm)']]
y_i = (y_i == 2)
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_i,y_i, test_size=0.2,random_state=42)


# In[12]:


svm_ns_iris = LinearSVC(C=1, loss = "hinge", random_state=42)
svm_clf_iris = Pipeline([('scaler', StandardScaler()),
                         ('linear_svc', LinearSVC(C=1, loss='hinge', random_state=42))])

svm_ns_iris.fit(X_train_iris,y_train_iris)
svm_clf_iris.fit(X_train_iris,y_train_iris)


# In[13]:


no_scale_acc_train_i = accuracy_score(y_train_iris, svm_ns_iris.predict(X_train_iris))
no_scale_acc_test_i = accuracy_score(y_test_iris, svm_ns_iris.predict(X_test_iris))

scale_acc_train_i = accuracy_score(y_train_iris, svm_clf_iris.predict(X_train_iris))
scale_acc_test_i = accuracy_score(y_test_iris, svm_clf_iris.predict(X_test_iris))


# In[14]:


iris_acc = [no_scale_acc_train_i, no_scale_acc_test_i, scale_acc_train_i, scale_acc_test_i]
iris_acc


# In[15]:


with open('iris_acc.pkl', 'wb') as f:
    pickle.dump(iris_acc,f)


# In[ ]:




