#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# data preparation


import numpy as np
import pandas as pd
 infy= pd.read_csv('data.csv')
    
    
    
    


# In[ ]:


infy.head()


# In[ ]:


infy.tail()


# In[ ]:


#df.count
print(infy.count()
      
      
print(infy["close Price"].count())
      
      


# In[ ]:


#df.min()
print(infy["close price"].min())


# In[ ]:


#df.max()
print(infy["close price"].max())


# In[ ]:


# mean median mode
print(infy["close price"].mean())
print(infy["close price"].median())
print(infy["close price"].mode())


# In[ ]:


#sum diff pct change

print(infy['total traded qty'].sum())


print(infy["close price"].diff())


# In[ ]:


print(infy["close price"].pct_change())


# In[ ]:


#visualising

import matplolib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('seaborn-darkgrid')

plt.figure(figsize=(10,5))

plt.ylabel('daily returns of infosys')
infy_["close price "].pct_change().plot()
plt.show()


# In[ ]:


#var , std  , moving average
print(infy["close price"].var())
print(infy["close price"].std()
      
infy["close price"].rolling(window= 20).mean().plot()
      


# In[ ]:


#visualise ma
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=10,5))
plt.ylabel('closing price')

#for another example

infy["close Price"].rolling(window= 20).mean().plot()

infy["close price"].plot()
plt.show()


# In[ ]:


#df expanding min

print(infy["close price"].expanding(min_periods= 20).mean())


# In[ ]:


import matplolib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize = (10,5))
plt.ylabel('daily returns of infy')

infy["close price"].expanding(min_period=20).mean.plot()

infy["close price"].plot()

plt.show()


# In[ ]:


# to fing cov we have t also load another data ste  of let say tcs


#syntax for cov

print(infy["close price'].cov(tcs["close price"]))


# In[ ]:


print(infy["close price"].corr(tcs["close price"]))


# In[ ]:


#kurtosis skew
print(infy["close price"].kurt())
print(infy["close price"].skew())


# In[ ]:


#visua;lise both distributions

import seaborn as sns
sns.set(color_codes = True)

sns.displot(infy["close price"])

plt.show()


# In[ ]:


import seaborn as sns
sns.set(color_codes = false)

sns.displot(tcs["close price"])
plt show()

