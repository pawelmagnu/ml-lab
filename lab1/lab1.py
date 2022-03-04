#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, tarfile, urllib, gzip
os.system('mkdir data')
os.system('curl -O https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz')
os.system('tar xfz housing.tgz')
os.system('gzip housing.csv')
os.system('rm housing.tgz')
os.system('mv housing.csv.gz data')
os.system('gzcat data/housing.csv.gz | head -4')


# In[2]:


import pandas as pd
df = pd.read_csv('data/housing.csv.gz')
print(df.head())
print(df.info())


# In[3]:


df['ocean_proximity'].value_counts()


# In[4]:


df['ocean_proximity'].describe()


# In[5]:


import matplotlib.pyplot as plt    # potrzebne ze względu na argument cmap
df.hist(bins=50, figsize=(20,15))
plt.savefig('obraz1.png')


# In[6]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.1, figsize=(7,4))
plt.savefig('obraz2.png')


# In[7]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.4, figsize=(7,3), colorbar=True,
        s=df["population"]/100, label="population", 
        c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig('obraz3.png')


# In[8]:


df.corr()["median_house_value"].sort_values(ascending=False).reset_index().rename(columns={'index':'atrybut','median_house_value':'wspolczynnik_korelacji'}).to_csv('korelacja.csv',index=False)


# In[9]:


import seaborn as sns
sns.pairplot(df)


# In[10]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df,
                                       test_size=0.2,
                                       random_state=42)
len(train_set),len(test_set)


# In[11]:


train_set.head()


# In[12]:


test_set.head()


# In[13]:


df.corr()["median_house_value"].sort_values(ascending=False)


# In[14]:


train_set.corr()["median_house_value"].sort_values(ascending=False)


# In[15]:


test_set.corr()["median_house_value"].sort_values(ascending=False)


# In[16]:


train_set.to_pickle('train_set.pkl')
test_set.to_pickle('test_set.pkl')


# In[ ]:




