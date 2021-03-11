#!/usr/bin/env python
# coding: utf-8

# In[5]:


#if else sttements
stock_price_tesla = 500
if (stock_price_tesla <1000 ):
    print("we will buy 500 shares of tesla")
    
elif(stock_price_tesla == 1000):
    print("we will buy 200  shares of tesla ")
    
elif(stock_price_tesla > 1000):
    print("we will buy 100 shares of tesla")
    
    


# In[ ]:


#for loop


# In[3]:


close_price_tesla = [300, 500, 600, 700, 800, 1000]
for i in close_price_tesla :
    if i <300 :
        print("we buy ")
    if i == 300 :
        print("no position")
    if i >300 :
        print("we sell")
        
print("we are out of loop")


# In[9]:


stock_price_tesla = 300

if(stock_price_tesla > 500):
    print("we will sell stock & book profit")
else:
    print("we will keep buying the stock")
          
   


# In[ ]:


#another examplr

import numpy as np
import pandas as pd
infy= pd.read_csv('data.csv')
infy


# In[10]:


#we will take close price  for 'for loop
for i in range(len(infy)):
    if(infy.iloc[i]["close price"]<1120):
        print("we buy")
        
        
    elif((infy.iloc[i]["cose Price"]>1120) & (infy.iloc[i]["close Price"]<1150)):
        
        print("we do nothing")
        
        
        
    elif(infy.iloc[i]["close price"]> 1150):
        
        print("we sell")
        
        


# In[11]:


#while loop
a= 0
while a <= 10:
    a = a+1
    print(a)
print("we are out of loop")


# In[ ]:




