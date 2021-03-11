#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


#matplotlib & seaborn are used for plotting graphs
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matploltib', 'inline')
plt.style.use('seaborn-darkgrid')


# In[ ]:


import yfinance as yf


# In[ ]:


#read data
df = yf.dowmnload('GLD',"2008-01-01-", '2020-6-22' , auto_adjust = True)


# In[ ]:


#only keep close columns
df = df[['Close']


# In[ ]:


#drop rows with missing values
df = df.drop()


# In[ ]:


#plot the closing price of gold
df.close.plot(figsize=(10,7),color = 'r')
plt.ylabel("Gold etf price ")
plt.title("gold etf price series")
plt.show()


# In[ ]:


#define explanatory variables
df['s_3']= df['close].rolling(window=3).mean()
df['s_9']=df['close'].rolling(window=9).mean()
df['next_day_price']= df['close'].shift(-1)


# In[ ]:


df = df.dropna()
x= df[['s_3',"s_9"]]


# In[ ]:


#define target variable
y= df['next_day_price']


# In[ ]:


#split the data into train &test
t=.8
t = int(t*len(df))

#train dataset
X_train = X[:t]
Y_train = Y[:t]


# In[ ]:


#test datset
X_test = X[t:]
Y_test = Y[t:]


# In[ ]:


#linear reg
#y= m1*x1 +m2*x2 + C
#gold etf price = m1 * 3 days movinf average + m2 * 15 days moving average + c


# In[ ]:


linear =  LinearRegresion().fit(X_train , y_train)
print("Linear Regression model ")
print(" gold etf price (y) = % .2f * 3 days moving average (x1) \ 
+%.2f*9 days moving average(x2)\ + %.2f(constant) " % (linear.coef_[0], linear.coef_[1], linear.intercept_))


# In[ ]:


output
#predictin the gold etf prices
predicted_price = linear.predict(X_test)
predicted_price = pd.dataframe(predicted_price , index = y_test , columns = [' price'])
predicted_price.plot(figsize= (10,7))
y_test.plot(
plt.legend(['predicted_price',"actual_price"])
plt.ylabel("gold etf price")
plt.show


# In[ ]:


#compute goodness of fitusing score function
# R SQUARE
r2_score = linear.score(x[t:],y[t:])*100
float("{0:.2f}".format(r2_score))

