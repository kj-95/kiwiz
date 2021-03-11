#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.svm import svc
from sklarn.metrics import accuracy_score


# In[ ]:


#for data manipulation
import pandas as pd
import numpy as np


# In[ ]:


#to plot
import matplotlib.pyplot as plt
plt.style.use('seaborn=darkgrid')


# In[ ]:


#read the csv files
df = pd.read_csv(path + 'spy.csv', index =_col = 0)


# In[ ]:


#conver index to date time format
df.index =pd.to_datetime(df.index)


# In[ ]:


#print first 5 rows 
df.head()


# In[ ]:


#define explanoatory variables
#create predictor variables
df['open-close']=df.open-df.close
df['high-low']=df.high- df.low


# In[ ]:


#store all predictor variables in x
x= df[['open-close','high-low']]
x.head()


# In[ ]:


#define target variable
y=np.where(df['close'].shift(-1) > df['close'],1,0)


# In[ ]:


#split the data into train & test
split = int(split_percentage*len(df))

#train
x_train= x[split:]
y_train= Y[split:]

#test
x_test= x[:split]
y_test= Y[:split]


# In[ ]:


#svc
from sklearn.svm import svc
svc().fit(X,Y)


# In[ ]:


#support vector cla
cls= SVC().fix(X_train , y_train)


# In[ ]:


#classifier accuracy
from sklearn.metrics import accuracy_score
accuracy_scorey(y_true, y_predicted)


# In[ ]:


#train and test accuracy
accuracy_train = accuracy_score(y_train , cls.predict(X_train))
accuracy_test = accuracy_score(y_test, cls.predict(X_test))

print('\nTrain Acurracy:{: .2f} %'.format(accuracy_train*100))
print('Test Accuracy ; { : .2f }%.format(accuracy_test*100
# In[ ]:


#predicted signal
df['predicted_Signal']=cls.predict(x)


# In[ ]:


#calculate daily returns
df['Return'] = df.close.pct_change()

#calstrategy returns
df['strategy_return'] = df.return* df.predicted_signal.shift(1)
# In[ ]:


#cal geom retirns
geometric_returns = (df.strategy_return.iloc[slit:] +1).cumprod()


#plot geometric returns
geometric_returns.plot(figsize= (10,7), color = "8")
plt.ylabel("strategy returns (%)")
plt.xlabel("date")
plt.show

