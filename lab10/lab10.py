#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


# In[2]:


X_train = X_train/255
X_test = X_test/255
X_valid = X_test[:int(len(X_test)*0.1),:]
y_valid = y_test[:int(len(y_test)*0.1)]


# In[3]:


# plt.imshow(X_train[142], cmap="binary")
# plt.axis('off')
# plt.show()


# In[4]:


class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
"sandał", "koszula", "but", "torba", "kozak"]
# class_names[y_train[142]]


# In[5]:


def get_callback(filename):
    root_logdir = os.path.join(os.curdir, filename)
    def get_run_logdir():
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)
    run_logdir = get_run_logdir()
    return tf.keras.callbacks.TensorBoard(run_logdir)


# In[6]:


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation='softmax'))
optimizer = keras.optimizers.SGD(lr=0.1)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])


# In[7]:


history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid),
                    callbacks=[get_callback("image_logs")])


# In[8]:


# image_index = np.random.randint(len(X_test))
# image = np.array([X_test[image_index]])
# confidences = model.predict(image)
# confidence = np.max(confidences[0])
# prediction = np.argmax(confidences[0])
# print("Prediction:", class_names[prediction])
# print("Confidence:", confidence)
# print("Truth:", class_names[y_test[image_index]])
# plt.imshow(image[0], cmap="binary")
# plt.axis('off')
# plt.show()


# In[9]:


# %load_ext tensorboard


# In[10]:


# %tensorboard --logdir ./housing_logs


# In[11]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()


# In[12]:


X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data,housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,y_train_full, random_state=42)


# In[13]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# In[14]:


es = tf.keras.callbacks.EarlyStopping(patience=5,min_delta=0.01,verbose=1)


# In[15]:


model2 = keras.models.Sequential()
model2.add(keras.layers.Dense(30, activation='relu',input_shape=X_train.shape[1:]))
model2.add(keras.layers.Dense(1))
model2.compile(loss='mse',optimizer='sgd')


# In[16]:


history2 = model2.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid),
                    callbacks=[get_callback("housing_logs"),es])


# In[17]:


model3 = keras.models.Sequential()
model3.add(keras.layers.Dense(30, activation='relu',input_shape=X_train.shape[1:]))
model3.add(keras.layers.Dense(5))
model3.add(keras.layers.Dense(1))
model3.compile(loss='mse',optimizer='sgd')

history3 = model3.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid),
                    callbacks=[get_callback("housing_logs"),es])


# In[18]:


model4 = keras.models.Sequential()
model4.add(keras.layers.Dense(30, activation='relu',input_shape=X_train.shape[1:]))
model4.add(keras.layers.Dense(5, activation='sigmoid'))
model4.add(keras.layers.Dense(1))
model4.compile(loss='mse',optimizer='sgd')

history4 = model4.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid),
                    callbacks=[get_callback("housing_logs"),es])


# In[19]:


model.save('fashion_clf.h5')
model2.save('reg_housing_1.h5')
model3.save('reg_housing_2.h5')
model4.save('reg_housing_3.h5')


# In[ ]:




