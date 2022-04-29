#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# In[2]:


from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics import silhouette_score,confusion_matrix
import pickle


# In[3]:


k_s = [8,9,10,11,12]
kmeans_sil = []
kmeans_clf = []
y_pred_kmeans = []
for k in k_s:
    kmeans = KMeans(n_clusters=k,random_state=42)
    y_pred = kmeans.fit_predict(X)
    kmeans_sil.append(silhouette_score(X, kmeans.labels_))
    kmeans_clf.append(kmeans)
    y_pred_kmeans.append(y_pred)


# In[4]:


# kmeans_sil


# In[5]:


# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(k_s,kmeans_sil)


# In[6]:


conf_mat = confusion_matrix(y,y_pred_kmeans[2])


# In[7]:


# conf_mat


# In[8]:


kmeans_argmax = set({})
for row in conf_mat:
    kmeans_argmax.add(np.argmax(row))


# In[9]:


kmeans_argmax = list(kmeans_argmax)


# In[10]:


dist = []
for index, x1 in enumerate(X[:300]):
    for index2,x2 in enumerate(X[index+1:]):
        dist.append(np.linalg.norm(x1-x2))


# In[11]:


dist = np.sort(dist)[:10]


# In[12]:


# dist


# In[13]:


s = np.average(dist[:3])


# In[14]:


esys = np.arange(s,1.1*s,0.04*s)


# In[15]:


# esys


# In[16]:


dbs = []
for epsy in esys:   
    dbscan = DBSCAN(eps=epsy)
    dbscan.fit(X)
    dbs.append(dbscan)


# In[17]:


dbscan_len = []
for dbsc in dbs:
    dbscan_len.append(len(list(set(dbsc.labels_))))


# In[18]:


# dbscan_len


# In[19]:


with open('kmeans_sil.pkl','wb') as f:
    pickle.dump(kmeans_sil,f)


# In[20]:


with open('kmeans_argmax.pkl','wb') as f:
    pickle.dump(kmeans_argmax,f)


# In[21]:


with open('dist.pkl','wb') as f:
    pickle.dump(dist,f)


# In[22]:


with open('dbscan_len.pkl','wb') as f:
    pickle.dump(dbscan_len,f)

