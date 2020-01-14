#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn.datasets
import numpy as np


# In[2]:


cancer_ds = sklearn.datasets.load_breast_cancer()


# In[3]:


X = cancer_ds.data
Y = cancer_ds.target


# In[4]:


print(X)
print(Y)


# In[5]:


print(X.shape, Y.shape)


# In[6]:


import pandas as pd


# In[7]:


data = pd.DataFrame(cancer_ds.data, columns=cancer_ds.feature_names)


# In[8]:


data['class'] = cancer_ds.target


# In[9]:


data.head(10)


# In[10]:


data.describe()


# In[11]:


print(data['class'].value_counts)


# In[12]:


print(data['class'].value_counts())


# In[13]:


print(cancer_ds.target_names)


# In[14]:


data.groupby('class').mean()


# In[15]:


data.groupby('class').min()


# In[16]:


data.groupby('class').max()


# # Train test split

# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X = data.drop('class', axis =1)
Y = data['class']


# In[19]:


data.describe()


# In[20]:


type(X)


# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y)


# In[22]:


print(Y.shape, Y_train.shape, Y_test.shape)


# In[23]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.1)


# In[24]:


print(Y.mean(), Y_train.mean(), Y_test.mean())


# In[25]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify = Y)


# In[26]:


print(X.mean(), X_train.mean(), X_test.mean())


# In[27]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify = Y, random_state =1)


# In[28]:


print(X.mean(), X_test.mean(), X_train.mean())


# In[29]:


import matplotlib.pyplot as plt


# In[30]:


plt.plot(X_test.T, '.')
plt.xticks(rotation ='vertical')
plt.show()


# In[31]:


X_binarised_train = X_train.apply(pd.cut, bins=2, labels=[1,0])


# In[32]:


plt.plot(X_binarised_train.T, '.')
plt.xticks(rotation ='vertical')
plt.show()


# In[33]:


X_binarised_test = X_test.apply(pd.cut, bins=2, labels=[1,0])


# In[34]:


type(X_binarised_test)


# In[35]:


X_binarised_test = X_binarised_test.values
X_binarised_train = X_binarised_train.values


# In[36]:


type(X_binarised_test)


# In[66]:


from random import randint
b = 3
i = randint(0, X_binarised_train.shape[0])

print('For row', i)

if (np.sum(X_binarised_train[i, :]) >= b):
  print('MP Neuron inference is malignant')
else:
  print('MP Neuron inference is benign')
  
if (Y_train[i] == 1):
  print('Ground truth is malignant')
else:
  print('Ground truth is benign')


# In[96]:


b = 30
Y_pred_train = []
accurate_rows = 0

for x, y in zip(X_binarised_train, Y_train):
  y_pred = (np.sum(x) >= b)
  Y_pred_train.append(y_pred)
  accurate_rows += (y == y_pred)
  
print(accurate_rows, accurate_rows/X_binarised_train.shape[0])


# In[69]:


for b in range(X_binarised_train.shape[1] + 1):
  Y_pred_train = []
  accurate_rows = 0

  for x, y in zip(X_binarised_train, Y_train):
    y_pred = (np.sum(x) >= b)
    Y_pred_train.append(y_pred)
    accurate_rows += (y == y_pred)

  print(b, accurate_rows/X_binarised_train.shape[0])  


# In[72]:


from sklearn.metrics import accuracy_score


# In[117]:


b = 26
Y_pred_test = []
for x in X_binarised_test:
  y_pred = (np.sum(x) >= b)
  Y_pred_test.append(y_pred)

accuracy = accuracy_score(Y_pred_test, Y_test)
print(b, accuracy)  

