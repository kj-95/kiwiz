#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#syntax for new dict

new_dict = ()
type(new_dict)


# In[7]:


#creating new 
new_dict ={'jack':25000,'Rose': 8965,'keval':9500}
type(new_dict)


# In[ ]:


#print the dictionary
print(new_dict)


# In[4]:


#print value of particular key
new_dict['jack']


# In[9]:


#print multiple values of various keys
new_dict['jack'], new_dict['Rose']


# In[ ]:


#dictionary manipulations
#to know no of key :value pairs in dict
len(x_dict)


# In[10]:


print(new_dict)
len(new_dict)


# In[11]:


# x_dict.values gives values of dict

new_dict.values()


# In[ ]:


#del statement for deleting any keys from dict


del new_dict['keval']
print(new_dict)


# In[12]:


#x_dict .pop pops a value of key
new_dict.pop('jack')


# In[13]:


#sort(x_dict) sorting by values
sorted(new_dict)


# In[14]:


#clears content of dict

new_dict.clear()
print(new_dict)


# In[ ]:




