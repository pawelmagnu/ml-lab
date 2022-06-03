#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import tensorflow as tf
import os
import time
import numpy as np
import pickle
from sklearn.model_selection import RandomizedSearchCV
import scikeras
from scikeras.wrappers import KerasRegressor


# In[2]:


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()


# In[3]:


es = tf.keras.callbacks.EarlyStopping(patience=10,min_delta=1.0,verbose=1)


# In[4]:


def get_callback(filename,name,value):
    root_logdir = os.path.join(os.curdir, filename)
    ts = int(time.time())
    filen = str(ts)+'_'+str(name)+'_'+str(value)
    return tf.keras.callbacks.TensorBoard(os.path.join(root_logdir, filen))


# In[5]:


def build_model(n_hidden, n_neurons, optimizer, learning_rate, momentum=0):
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Input(shape=13))
    for i in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
    if optimizer == 'sgd': 
        opt = keras.optimizers.SGD(lr=learning_rate,nesterov=False)
    elif optimizer == 'nesterov':
        opt = keras.optimizers.SGD(lr=learning_rate,nesterov=True)
    elif optimizer == 'momentum':
        opt = keras.optimizers.SGD(lr=learning_rate,nesterov=False,momentum=momentum)
    elif optimizer == 'adam':
        opt = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='mse',optimizer=opt,metrics=['mse','mae'])
    return model


# In[6]:


tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


# In[7]:


learning = [0.000001,0.00001,0.0001]
lr = []
for rate in learning:
    model = build_model(1,25,'sgd',rate)
    history = model.fit(X_train,y_train,epochs=100,validation_split=0.1,callbacks=[get_callback('tb_logs','lr',rate),es],verbose=0)
    result = (rate, history.history['loss'][-1], history.history['mae'][-1])
    lr.append(result)
lr


# In[8]:


hidden = [0,1,2,3]
hl = []
for hid in hidden:
    model = build_model(hid,25,'sgd',1e-05)
    history = model.fit(X_train,y_train,epochs=100,validation_split=0.1,callbacks=[get_callback('tb_logs','hl',hid),es],verbose=0)
    result = (hid, history.history['loss'][-1], history.history['mae'][-1])
    hl.append(result)
hl


# In[9]:


neurons = [5,25,125]
nn = []
for neuron in neurons:
    model = build_model(1,neuron,'sgd',1e-05)
    history = model.fit(X_train,y_train,epochs=100,validation_split=0.1,callbacks=[get_callback('tb_logs','nn',neuron),es],verbose=0)
    result = (neuron, history.history['loss'][-1], history.history['mae'][-1])
    nn.append(result)
nn


# In[10]:


opti = ['sgd','nesterov','momentum','adam']
opt = []
for optimizer in opti:
    model = build_model(1,25,optimizer,1e-05,0.5)
    history = model.fit(X_train,y_train,epochs=100,validation_split=0.1,callbacks=[get_callback('tb_logs','opt',optimizer),es],verbose=0)
    result = (optimizer, history.history['loss'][-1], history.history['mae'][-1])
    opt.append(result)
opt


# In[11]:


moment = [0.1,0.5,0.9]
mom = []
for momentum in moment:
    model = build_model(1,25,'momentum',1e-05,momentum)
    history = model.fit(X_train,y_train,epochs=100,validation_split=0.1,callbacks=[get_callback('tb_logs','mom',momentum),es],verbose=0)
    result = (momentum, history.history['loss'][-1], history.history['mae'][-1])
    mom.append(result)
mom


# In[12]:


with open('lr.pkl','wb') as f:
    pickle.dump(lr,f)
with open('hl.pkl','wb') as f:
    pickle.dump(hl,f)
with open('nn.pkl','wb') as f:
    pickle.dump(nn,f)
with open('opt.pkl','wb') as f:
    pickle.dump(opt,f)
with open('mom.pkl','wb') as f:
    pickle.dump(mom,f)


# In[13]:


param_distribs = {
"model__n_hidden": [0,1,2,3],
"model__n_neurons": [5,25,125],
"model__learning_rate": [0.000001,0.00001,0.0001],
"model__optimizer": ['sgd','nesterov', 'momentum','adam'],
"model__momentum": [0.1,0.5,0.9]
}


# In[14]:


keras_reg = KerasRegressor(build_model, callbacks=[es])


# In[15]:


rnd_search_cv = RandomizedSearchCV(keras_reg,param_distribs,n_iter=30,cv=3,verbose=1)
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_split=0.1)


# In[16]:


rnd_search_cv.best_params_


# In[17]:


with open('rnd_search.pkl','wb') as f:
    pickle.dump(rnd_search_cv.best_params_,f)


# In[ ]:




