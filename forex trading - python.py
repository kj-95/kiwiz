#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#define currency list
#ziploine library
from zipline.api import symbol
def initialize(context):
    context.currency_list=[symbol('fxcm: eur/usd'), symbol('fxcm : usd/jpy')]


# In[ ]:


#schedule a strategy function
schedule_function(strategy , date_rules.week_start, time_rules.market_open(minutes = 15))


# In[ ]:


from zipline.api import (symbol, schedule_function , date_rules,time_rules)

def initialise(context):
    schedule_function(strategy ,date_rules.week_start(),
                     time_rules.market_open(minutes = 15))


# In[ ]:


#fetchinf price of pairs

def strategy (context , date):
    currency_data = data.history(assets= context.currency_list , fields = ' price', bar_count= 50),
    frequency = 'id')
    print(currency_data.head(3))


# In[ ]:


#compute 50- day return


currency_returns = currency_data.iloc[-1]/currency_data.iloc[0]-1
print(currency_returns.head(3))


# In[ ]:




