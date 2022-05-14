#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
iris = load_iris(as_frame=True)


# In[2]:


y_0 = (iris.target == 0).astype(int)
y_1 = (iris.target == 1).astype(int)
y_2 = (iris.target == 2).astype(int)


# In[3]:


perceptrons = []
accuracy_train = []
accuracy_test = []
coefs = []
for y in [y_0, y_1, y_2]:  
    X_train, X_test, y_train, y_test = train_test_split(iris.data, y, test_size=0.2, random_state=42)
    per_clf = Perceptron()
    per_clf.fit(X_train,y_train)
    accuracy_train.append(accuracy_score(y_train,per_clf.predict(X_train)))
    accuracy_test.append(accuracy_score(y_test,per_clf.predict(X_test)))
    perceptrons.append(per_clf)
    coefs.append(per_clf.coef_)


# In[4]:


per_acc = []
for z in zip(accuracy_train, accuracy_test):
    per_acc.append(z)


# In[5]:


per_wght = [(coefs[0][0][i],coefs[1][0][i],coefs[2][0][i]) for i in range(len(coefs[0][0]))]


# In[6]:


X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
y = np.array([0,1,1,0])


# In[ ]:


optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.1)
mlp_xor_weights = []
while True:
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2, input_dim=2, activation ='sigmoid'))
    model.add(keras.layers.Dense(1, activation = 'sigmoid'))
    model.compile(loss='mse',
                  optimizer=optimizer_adam,
                  metrics=['binary_accuracy'])
    history = model.fit(X, y, epochs=1000, verbose=False)
    y_pred = model.predict(X)
    if y_pred[0] < 0.1 and y_pred[3] < 0.1 and y_pred[1] > 0.9 and y_pred[2] > 0.9:
        mlp_xor_weights = model.get_weights()
        break


# In[ ]:


# per_acc


# In[ ]:


# per_wght


# In[ ]:


# mlp_xor_weights


# In[ ]:


with open('per_acc.pkl','wb') as f:
    pickle.dump(per_acc,f)
    
with open('per_wght.pkl','wb') as f:
    pickle.dump(per_wght,f)
    
with open('mlp_xor_weights.pkl','wb') as f:
    pickle.dump(mlp_xor_weights,f)

