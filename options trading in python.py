#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import libraries
import numpy as np
import matplotlib.pyplot as plt
#for making attractive and informative statistical graph
plt.style.use('seaborn-darkgrid')


# In[8]:


def put_payoff(st, strike_price, premium):
    pnl = np.where(st<strike_price , strike_price - st , 0)
    return pnl - premium


# define parameters

# In[9]:


#infosys stock price 
spot_price =900
#put stike price & cost
strike_price = 900
premium = 20


# In[18]:


#put option buyer payoff


payoff_long_put = put_payoff(strike_price, premium)
#plot the graph
fig, ax = plt.subplots(figsize=(8,5))
ax.spines['bottom'].set_position('zero')
ax.plot(st,payoff_long_put , label = ' put option buyer payoff')
plt.xlabel('infosys stock price ')
plt.ylabel('profit & loss')


# In[19]:


import numpy as np
import pandas as pd
#infosys stock price 
stock_price= 900
  
#call strike price & cost
strike_price = 900
premium = 20


# In[14]:


def call_payoff(st, strike_price, premium):
    pnl = np.where(st>strike_price , st - strike_price , 0)
    return pnl - premium


# In[15]:


#call option buyer payoff
payoff_long_call = call_payoff(st, strike_price , premium)
fig , ax = plt.subplots(figsize = (8,5))
ax.spines['bottom'].set_position('zer0')
ax.plot(st, payofff_long_call , lable = ' call option buyer payoff')
plt.xlabel('infosys stock price' )
plt.ylabel('profit and loss')
plt.legend()
plt.show()           
        


# In[ ]:


call option seller payoff
payoff_sport_call = payoff_long_call * -1.0
fig , ax = plt.subplots(figsize =(8,5))
ax.spines['bottom'].set_position('zero')
ax.plotst,payoff-short_call , label= ' short 940 strike call ' , color = 'r ')

plt.ylabel('infosys stock price')
plt.xlabel('profit & loss')
plt.legend()
plt.show()


# In[ ]:


#computing historical volatility
#import lib


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


#fetching stock data
data = pd.read_csv('/...//apple_stock_data')
data.head()


# In[ ]:


#computing log returns
data['log returns']= np.log(data['adj_close]'])


# In[20]:


#plot the volatility
plt.figure(figsize = (10,7))
plt.plot(data['20 day historical volatily'], color = 'b' , label = ' 20 day historical volatility')
plt.legend(loc = 'best')
plt.show()


# In[ ]:


#bull call spread
import numpy as np
import matplotlib.pyplot as plt
#for making an  attractive and informative statistical graph
plt.style.use ('seaborn darkgrid')


# In[ ]:


#call payoff
def call_payoff(st,strike_price, premium):
    return np.where(st> strike_price , sT - strike_price , 0)- premium


# In[ ]:


#define parameters
#nifty stock price
spot_price = 14500


#lon_call
strike_call_long_call = 14800
premium_long_call = 15


# In[ ]:


#short call
strike_price_short_call = 15000
premium_short_call = 10


# In[ ]:


#stock price range at expiration of call
st = np.arange(0.95*spot_price , 1.1*spot_price , 1)


# In[ ]:


#long call payoff
payoff_longcall = call_payoff(st, strike_price_long_call , premium_longcall)
   
fig , ax = plt.subplot(figsize = (8,5))
ax. spines['bottom'.set_position('zero')
ax.plot(sT , payoff_loncall , label = ' long 14000 strike call')
plt.xlabel('nifty price'  )
plt.ylabel('p&l')
           plt.legend
           plt.show


# In[21]:


#bull call payoff

payoff_bull_call_spread = payoff_long_call + payoff_short_call

print(" max profit", max( payoff_bull_call_spread))
print("max loss"), min(payoff_bull_call_spread)


#plot
fig , ax plt.subplot(figsize= (8,5))
ax.spines['bottom'].set_position('zero')
ax.plot(sT, payoff_long_call, '_ _ ', label = 'long 14800 strike call ' , color 'g')
ax.plot(sT , payoff_short_call , '_ _', label =' short 15000 strike call ' color =r
        
        
        ax.plot(st , payoff_short_call , '_ _', label = short 15000 strike call= 'r')
ax.plot(sT , payoff_bull_call_spread , label = ' bull call spread '


# In[22]:


plt.xlabel('nifty')
plt.ylabel('p&l')
plt.legend()
pltshow()


# In[ ]:


#nifty price
spot_price = 14500
#short call=
strike_price_short_call = 300
premium_shirt_call = 10

#stock price range at expiration
sT = np.arange(0.9*spot_price , 1.1*spot_price, 1)


# In[ ]:


payoff
payoff_wipro_stock = st -300

fig , ax = plt.subplot(figsize= (8,50))
ax .spines([' bottom'].set_position('zero')
ax.plot(sT , payoff_wipro_stock , label = ' wipro stock payoff' , color ='r') 
plt.xlabel('wipro stock')
plt.ylabel('p&l')
plt.show()
           

