#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.datasets import fetch_openml 
import numpy as np
mnist = fetch_openml('mnist_784', version=1)


# In[4]:


print((np.array(mnist.data.loc[42]).reshape(28, 28) > 0).astype(int))


# In[15]:


X,y = mnist.data,mnist.target


# In[16]:


y = y.sort_values()


# In[17]:


y.index


# In[18]:


X.reindex(y.index)


# In[19]:


X_train, X_test = X[:56000], X[56000:]
y_train, y_test = y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[22]:


y_train.describe()


# In[23]:


y_test.describe()


# In[24]:


# nie da sie osiagnac takiego podzialu. funkcja train_test_split, bo dokonuje ona pomieszania przed podzieleniem


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[27]:


y_train.describe()


# In[28]:


y_test.describe()


# In[47]:


y0_train = (y_train == '0')
y0_test = (y_test == '0')


# In[48]:


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
clf = SGDClassifier(max_iter=1000)


# In[49]:


clf.fit(X_train,y0_train)
y0_pred_train = clf.predict(X_train)
y0_pred_test = clf.predict(X_test)
acc_train = accuracy_score(y0_train,y0_pred_train)
acc_test = accuracy_score(y0_test,y0_pred_test)


# In[50]:


acc_train


# In[51]:


acc_test


# In[55]:


from sklearn.model_selection import cross_val_score
cva = cross_val_score(clf, X_train, y_train, cv=3)
acc = [acc_train, acc_test]


# In[60]:


import pickle
with open('sgd_acc.pkl', 'wb') as f:
    pickle.dump(acc,f)
with open('sgd_cva.pkl', 'wb') as f:
    pickle.dump(cva,f)


# In[52]:


clf2 = SGDClassifier(max_iter=1000)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)


# In[53]:


print(conf_mat)


# In[61]:


with open('sgd_cmx.pkl','wb') as f:
    pickle.dump(conf_mat, f)


# In[ ]:




