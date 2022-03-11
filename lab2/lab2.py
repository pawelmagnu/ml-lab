#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml 
import numpy as np
mnist = fetch_openml('mnist_784', version=1)


# In[2]:


print((np.array(mnist.data.loc[42]).reshape(28, 28) > 0).astype(int))


# In[3]:


X,y = mnist.data,mnist.target


# In[4]:


y = y.sort_values()


# In[5]:


y.index


# In[6]:


X.reindex(y.index)


# In[7]:


X_train, X_test = X[:56000], X[56000:]
y_train, y_test = y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[8]:


y_train.describe()


# In[9]:


y_test.describe()


# In[10]:


# nie da sie osiagnac takiego podzialu. funkcja train_test_split, bo dokonuje ona pomieszania przed podzieleniem


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


y_train.describe()


# In[13]:


y_test.describe()


# In[14]:


y0_train = (y_train == '0')
y0_test = (y_test == '0')


# In[15]:


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
clf = SGDClassifier(max_iter=1000)


# In[16]:


clf.fit(X_train,y0_train)
y0_pred_train = clf.predict(X_train)
y0_pred_test = clf.predict(X_test)
acc_train = accuracy_score(y0_train,y0_pred_train)
acc_test = accuracy_score(y0_test,y0_pred_test)


# In[17]:


acc_train


# In[18]:


acc_test


# In[20]:


from sklearn.model_selection import cross_val_score
cva = cross_val_score(clf, X_train, y0_train, cv=3)
acc = [acc_train, acc_test]


# In[22]:


import pickle
with open('sgd_acc.pkl', 'wb') as f:
    pickle.dump(acc,f)
with open('sgd_cva.pkl', 'wb') as f:
    pickle.dump(cva,f)


# In[23]:


clf2 = SGDClassifier(max_iter=1000)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)


# In[24]:


print(conf_mat)


# In[26]:


with open('sgd_cmx.pkl','wb') as f:
    pickle.dump(conf_mat, f)


# In[27]:


cva


# In[28]:


acc


# In[ ]:




