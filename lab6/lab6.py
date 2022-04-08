#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[2]:


data_breast = datasets.load_breast_cancer(as_frame=True)


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(data_breast.data,data_breast.target,test_size=0.2,random_state=42)


# In[4]:


X_train2 = X_train[['mean texture', 'mean symmetry']]
X_test2 = X_test[['mean texture', 'mean symmetry']]


# In[5]:


log_clf = LogisticRegression()
tree_clf = DecisionTreeClassifier()
knn_clf = KNeighborsClassifier()

hard_clf = VotingClassifier(
    estimators=[('lr', log_clf),
                ('tr', tree_clf),
                ('knn', knn_clf)],
    voting='hard')

soft_clf = VotingClassifier(
    estimators=[('lr', log_clf),
                ('tr', tree_clf),
                ('knn', knn_clf)],
    voting='soft')


# In[6]:


classifiers = [tree_clf,log_clf,knn_clf,hard_clf,soft_clf]
acc = []
for clf in classifiers:
    clf.fit(X_train2, y_train)
    first_val = accuracy_score(y_train, clf.predict(X_train2))
    second_val = accuracy_score(y_test, clf.predict(X_test2))
    acc.append((first_val,second_val))


# In[7]:


acc


# In[8]:


with open('acc_vote.pkl','wb') as f:
    pickle.dump(acc,f)


# In[9]:


with open('vote.pkl','wb') as f:
    pickle.dump(classifiers,f)


# In[10]:


bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                            bootstrap=True)
bag50_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                              max_samples=0.5, bootstrap=True)
pas_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                            bootstrap=False)
pas50_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                              max_samples=0.5, bootstrap=False)
rnd_clf = RandomForestClassifier(n_estimators=30)
ada_clf = AdaBoostClassifier(n_estimators=30)
gbc_clf = GradientBoostingClassifier(n_estimators=30)


# In[11]:


classif = [bag_clf,bag50_clf,pas_clf,pas50_clf,rnd_clf,ada_clf,gbc_clf]
accurac = []
for clf in classif:
    clf.fit(X_train2, y_train)
    first_val = accuracy_score(y_train, clf.predict(X_train2))
    second_val = accuracy_score(y_test, clf.predict(X_test2))
    accurac.append((first_val,second_val))


# In[12]:


accurac


# In[13]:


with open('acc_bag.pkl','wb') as f:
    pickle.dump(accurac,f)


# In[14]:


with open('bag.pkl','wb') as f:
    pickle.dump(classif,f)


# In[15]:


bagrnd_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=True, 
                               bootstrap_features=False, max_samples=0.5, max_features=2)
bagrnd_clf.fit(X_train,y_train)
fea_acc = [accuracy_score(y_train, bagrnd_clf.predict(X_train)),
           accuracy_score(y_test, bagrnd_clf.predict(X_test))]


# In[16]:


fea_acc


# In[17]:


with open('acc_fea.pkl','wb') as f:
    pickle.dump(fea_acc,f)


# In[18]:


with open('fea.pkl','wb') as f:
    pickle.dump([bagrnd_clf],f)


# In[19]:


# bagrnd_clf.estimators_features_


# In[20]:


# bagrnd_clf.estimators_


# In[21]:


df = pd.DataFrame({'train_acc': pd.Series(dtype='float'),
                   'test_acc':  pd.Series(dtype='float'),
                   'feat_list': pd.Series(dtype='object')})


# In[22]:


for index in range(len(bagrnd_clf.estimators_)):
    x_train = X_train.iloc[:,bagrnd_clf.estimators_features_[index]]
    x_test = X_test.iloc[:,bagrnd_clf.estimators_features_[index]]
    feat_names = [str(x) for x in x_train.columns]
    clf = bagrnd_clf.estimators_[index]
    clf.fit(x_train,y_train)
    train_acc = accuracy_score(y_train, clf.predict(x_train))
    test_acc = accuracy_score(y_test, clf.predict(x_test))
    df.loc[len(df)] = [train_acc,test_acc,feat_names]


# In[23]:


df = df.sort_values(by=['train_acc','test_acc'],ascending=False)


# In[24]:


df


# In[25]:


with open('acc_fea_rank.pkl','wb') as f:
    pickle.dump(df,f)


# In[ ]:




