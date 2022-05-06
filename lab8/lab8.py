#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_iris
import numpy as np
import pickle


# In[2]:


data_breast_cancer = load_breast_cancer()
data_iris = load_iris()


# In[3]:


pca_breast = PCA(n_components=0.9)
pca_iris = PCA(n_components=0.9)
breast_pca = pca_breast.fit_transform(data_breast_cancer.data)
iris_pca = pca_iris.fit_transform(data_iris.data)


# In[4]:


# pca.explained_variance_ratio_


# In[5]:


# pca_iris.explained_variance_ratio_


# In[6]:


scaler_breast = StandardScaler()
scaler_iris = StandardScaler()


# In[7]:


pca_breast_t = PCA(n_components=0.9)
pca_iris_t = PCA(n_components=0.9)
breast_pca_t = pca_breast_t.fit_transform(scaler_breast.fit_transform(data_breast_cancer.data))
iris_pca_t = pca_iris_t.fit_transform(scaler_iris.fit_transform(data_iris.data))


# In[8]:


pca_bc = pca_breast_t.explained_variance_ratio_
pca_ir = pca_iris_t.explained_variance_ratio_


# In[9]:


# pca_bc


# In[10]:


# pca_ir


# In[11]:


idx_bc = list()
for comp in pca_breast_t.components_:
    idx_bc.append(np.argmax(np.abs(comp)))


# In[12]:


# idx_bc


# In[13]:


idx_ir = list()
for comp in pca_iris_t.components_:
    idx_ir.append(np.argmax(np.abs(comp)))


# In[14]:


# idx_ir


# In[15]:


with open('pca_bc.pkl','wb') as f:
    pickle.dump(pca_bc,f)


# In[16]:


with open('pca_ir.pkl','wb') as f:
    pickle.dump(pca_ir,f)


# In[17]:


with open('idx_bc.pkl','wb') as f:
    pickle.dump(idx_bc,f)


# In[18]:


with open('idx_ir.pkl','wb') as f:
    pickle.dump(idx_ir,f)


# In[ ]:




