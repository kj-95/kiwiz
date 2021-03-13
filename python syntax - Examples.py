#!/usr/bin/env python
# coding: utf-8

# In[6]:


#to check a person is eligible to vote or not

#get user name
print('enter your name:')
name=input()

#get the age

print('enter your age :')
age = int(input())

#condition to check if user is eligible to vate
if ( age >=18):
    print(name,'is eligible to vote')
else:
    print(name, 'is not eligible to  vote')


# type of syntax structures in python
# list structure
# docstrings
# quotation
# variables
# comments
# indentation
# identifiers
# stringformatters
# 

# In[8]:


#multiline statements
#use backward slash
print("Hihow are you?")


# In[ ]:


def func():
    """ This function prints out a greeting"""
    print("HI")
    func()


# In[9]:


#python indentaion
if 2>1:
    print("2 is the biggest person")
    print("but 1 is worth too")


# In[10]:


#quotations if u begin with single quote then end with single quote 
print('we need a mercedes')


# In[11]:


#variables
x=10
print(x)


# In[ ]:


#string formatters
x= 10; printer = "HP"
print("i just printed %s pages to the printer %s" % ( x , printer))


# In[16]:


#format method
x = 10, printer ='HP'
print('i just printed (0) pages to the printer(1)".format(x, printer))


# In[ ]:


#f -strings
print(f"I just printed(x) pages to the printer(printer)")


# some rules to follow while choosing an identifier
# 
# - it may only begin with A-Z , a-z or an underscore(_)
# - this may be followed by letters , digits , underscores
# - pyhton is case sensitive , Name & name aret diffferent  identifiers
# 
# 
# 

# some naming conventions that you should follow while using python
# 
# - use uppercase initials for class names , lowercase for others
# - name a private identifier with a leading underscore(_username)
# - name a stringly private identifier with two leading underscores(_password)
# - special identifiers by pytho end with two leading underscores
# 
# 
