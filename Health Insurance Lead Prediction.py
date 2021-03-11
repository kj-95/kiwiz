#!/usr/bin/env python
# coding: utf-8

# In[52]:


#Reading and Understanding the Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use('seaborn-deep')
plt.style.use('fivethirtyeight')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.figsize'] = (12, 8)

pd.options.mode.chained_assignment = None
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 400)
import warnings
warnings.filterwarnings('ignore')
import sklearn.base as skb
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import sklearn.utils as sku
import sklearn.linear_model as sklm
import sklearn.neighbors as skn
import sklearn.ensemble as ske
import scipy.stats as sstats
import random
seed = 12
np.random.seed(seed)

from datetime import date


# In[4]:


import catboost as cb


# In[1]:


get_ipython().system('pip install pandas-profiling')
import pandas_profiling as pp


# In[2]:


# important funtions
def datasetShape(df):
    rows, cols = df.shape
    print("The dataframe has",rows,"rows and",cols,"columns.")
    
# select numerical and categorical features
def divideFeatures(df):
    numerical_features = df.select_dtypes(include=[np.number])
    categorical_features = df.select_dtypes(include=[np.object])
    return numerical_features, categorical_features


# In[13]:


#read the file
df= pd.read_csv('C:/Users/user3/Downloads/train_Df64byy.csv')


# In[14]:


df.head


# In[16]:


df_test=pd.read_csv('C:/Users/user3/Downloads/test_YCcRUnU.csv')


# In[17]:


df_test.head


# In[18]:


# set target feature
targetFeature='Response'


# In[19]:


# check dataset shape
datasetShape(df)


# In[20]:


# remove ID from train data
df.drop(['ID'], inplace=True, axis=1)


# In[21]:


# check for duplicates
print(df.shape)
df.drop_duplicates(inplace=True)
print(df.shape)


# In[22]:


df.info()


# In[23]:


df_test.info()


# In[24]:


df.describe()


# EDA
# 

# In[25]:


cont_features, cat_features = divideFeatures(df)
cat_features.head()


# In[26]:


#Univariate Analysis
# check target feature distribution
df[targetFeature].hist()
plt.show()


# In[27]:


# boxplots of numerical features for outlier detection

fig = plt.figure(figsize=(16,16))
for i in range(len(cont_features.columns)):
    fig.add_subplot(3, 3, i+1)
    sns.boxplot(y=cont_features.iloc[:,i])
plt.tight_layout()
plt.show()


# In[28]:


# distplots for categorical data

fig = plt.figure(figsize=(16,20))
for i in range(len(cat_features.columns)):
    fig.add_subplot(3, 3, i+1)
    cat_features.iloc[:,i].hist()
    plt.xlabel(cat_features.columns[i])
plt.tight_layout()
plt.show()


# In[31]:


# plot missing values

def calc_missing(df):
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing != 0]
    missing_perc = missing/df.shape[0]*100
    return missing, missing_perc

if df.isna().any().sum()>0:
    missing, missing_perc = calc_missing(df)
    missing.plot(kind='bar',figsize=(16,6))
    plt.title('Missing Values')
    plt.show()
else:
    print("No missing values")


# In[32]:


sns.pairplot(df)
plt.show()


# In[33]:


# correlation heatmap for all features
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask = mask, annot=True)
plt.show()


# profiling whole data

# In[34]:


profile = pp.ProfileReport(df, title='Pandas Profiling Report', explorative=True)
profile.to_file("profile.html")


# In[35]:


profile.to_notebook_iframe()


# 3- data preparation

# In[36]:


#skewness
skewed_features = cont_features.apply(lambda x: x.skew()).sort_values(ascending=False)
skewed_features


# In[39]:


#handle missing
#plot missing value

def calc_missing(df):
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing != 0]
    missing_perc = missing/df.shape[0]*100
    return missing, missing_perc

if df.isna().any().sum()>0:
    missing, missing_perc = calc_missing(df)
    missing.plot(kind='bar',figsize=(14,5))
    plt.title('Missing Values')
    plt.show()
else:
    print("No Missing Values")


# In[40]:


# remove all columns having no values
df.dropna(axis=1, how="all", inplace=True)
df.dropna(axis=0, how="all", inplace=True)
datasetShape(df)


# In[ ]:


#Health Indicator Missing Prediction
# # convert city code to int after removing C from it
# df['City_Code'] = pd.to_numeric(df['City_Code'].map(lambda x:x[1:]))
# df_test['City_Code'] = pd.to_numeric(df_test['City_Code'].map(lambda x:x[1:]))
# df['City_Code'].head()


# In[41]:


cont_features, cat_features = divideFeatures(df)
cont_features.columns.tolist()


# In[47]:


# get all not null records for imputing
X_impute = df[df['Health Indicator'].isna()==False]
y_impute = X_impute.pop('Health Indicator')

# remove categorical cols and targetFeature from X_impute
X_impute = X_impute[cont_features.columns.tolist()]
X_impute.drop(['Holding_Policy_Type', targetFeature], inplace=True, axis=1)

# impute with CatBoostClassifier
pip install catboost
import catboost as cb
imputer_model = cb.CatBoostClassifier(random_state=seed, verbose=0)
imputer_model.fit(X_impute, y_impute)


# In[ ]:





# In[ ]:




