#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import pickle


# In[2]:


[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load("tf_flowers",
                                                            split=["train[:10%]", "train[10%:25%]", "train[25%:]"], as_supervised=True,
                                                            with_info=True)


# In[3]:


class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples


# In[4]:


# plt.figure(figsize=(12, 8))
# index = 0
# sample_images = train_set_raw.take(9) 
# for image, label in sample_images:
#     index += 1
#     plt.subplot(3, 3, index)
#     plt.imshow(image)
#     plt.title("Class: {}".format(class_names[label])) 
#     plt.axis("off")
# plt.show(block=False)


# In[5]:


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224]) 
    return resized_image, label


# In[6]:


batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1) 
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# In[7]:


# plt.figure(figsize=(8, 8)) 
# sample_batch = train_set.take(1)
# for X_batch, y_batch in sample_batch:
#     for index in range(12):
#         plt.subplot(3, 4, index + 1) 
#         plt.imshow(X_batch[index]/255.0)
#         plt.title("Class: {}".format(class_names[y_batch[index]])) 
#         plt.axis("off")
# plt.show()


# In[8]:


tf.config.run_functions_eagerly(True)
model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(224,224,  3)))
model.add(keras.layers.experimental.preprocessing.Rescaling(1./255))
model.add(keras.layers.Conv2D(32, (7, 7), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(n_classes, activation='softmax'))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'], )


# In[9]:


history = model.fit(train_set,
                    epochs=10,
                    validation_data=valid_set)


# In[ ]:


_, acc_train = model.evaluate(train_set)
_, acc_valid = model.evaluate(valid_set)
_, acc_test = model.evaluate(test_set)


# In[ ]:


with open('simple_cnn_acc.pkl','wb') as f:
    pickle.dump((acc_train,acc_valid,acc_test),f)


# In[ ]:


def preproces(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image) 
    return final_image, label


# In[ ]:


batch_size = 32
train_set = train_set_raw.map(preproces).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preproces).batch(batch_size).prefetch(1) 
test_set = test_set_raw.map(preproces).batch(batch_size).prefetch(1)


# In[ ]:


# plt.figure(figsize=(8, 8)) 
# sample_batch = train_set.take(1)
# for X_batch, y_batch in sample_batch:
#     for index in range(12):
#         plt.subplot(3, 4, index + 1)
#         plt.imshow(X_batch[index] / 2 + 0.5)
#         plt.title("Class: {}".format(class_names[y_batch[index]])) 
#         plt.axis("off")
# plt.show()


# In[ ]:


base_model = tf.keras.applications.xception.Xception(weights="imagenet", input_shape=[224, 224, 3], include_top=False)
for layer in base_model.layers: 
    layer.trainable = False


# In[ ]:


x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(5, activation='softmax')(x)
model = keras.Model(base_model.inputs,output)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


history = model.fit(train_set,
                    epochs=5,
                    validation_data=valid_set)


# In[ ]:


# for layer in base_model.layers: 
    # layer.trainable = True
# history = model.fit(train_set,
#                     epochs=10,
#                     validation_data=valid_set)


# In[ ]:


_, acc_train = model.evaluate(train_set)
_, acc_valid = model.evaluate(valid_set)
_, acc_test = model.evaluate(test_set)


# In[ ]:


with open('xception_acc.pkl','wb') as f:
    pickle.dump((acc_train,acc_valid,acc_test),f)


# In[ ]:




