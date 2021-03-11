#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.linear_model import logisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import lableEncoder
from sklearn.metrics import accuracy_score


# In[ ]:


#read dataset
data= pd.read_csv('Iris.csv')
print(data.head())


# In[ ]:


print('\n\nColumn names\n\n')
print(data.columns)


# In[ ]:


#label encode the target variable
encode = LableEncoder()
data.species = encode.fit_transform(data.species)


# In[ ]:


print(data.head())


# In[ ]:


#train-test-split
train,test = train_test_split(data,test_size=0.2, random_state=0)


# In[ ]:


print('shape of training data:',train.shape)
print('shape of testing data' , test.shape)


# In[ ]:


#sepearte the target and independent variable
train_x =test.drop(columns=['species'],axis=1)
train_y = test['species']


# In[ ]:


#create the object of model
model= logisticRegression()


# In[ ]:


model.fit(train_x, train_y)


# In[ ]:


predict=model.predict(test_x)


# In[ ]:


print('predicted values on test data'.encode.inverse_transform(predict))


# In[ ]:


print('\n\nAccuracy Score on test data:\n\n'
     print(accuracy_score(test_y, predict)))

