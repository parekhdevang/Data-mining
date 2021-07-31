#!/usr/bin/env python
# coding: utf-8

# In[3]:



# Imports
import math

import numpy as np
import pandas as pd
import scipy.stats

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns


# In[4]:


housingdf = pd.read_csv('AmesHousing.csv')
housingdf.head()


# In[5]:


print('Our training dataset has {} rows and {} columns.'.format(housingdf.shape[0], housingdf.shape[1]))


# In[12]:


#numerical & categorical columns 
catcols = housingdf.select_dtypes('object').columns
numcols = housingdf.select_dtypes('number').columns
print("Categorical variable count: ", len(catcols))
print("Numerical variable count: ", len(numcols))


# In[14]:


print("Categorical variables: ", catcols)
print("Numerical variables: ", numcols)


# ### EDA 

# In[11]:


print('The cheapest house sold for ${:,.0f} and the most expensive for ${:,.0f}'.format(
    housingdf.SalePrice.min(), housingdf.SalePrice.max()))
print('The average sales price is ${:,.0f}, while median is ${:,.0f}'.format(
    housingdf.SalePrice.mean(), housingdf.SalePrice.median()))
housingdf.SalePrice.hist(bins=75, rwidth=.8, figsize=(14,4))
plt.title('How expensive are houses?')
plt.show()


# In[16]:


# When were the houses built?
print('Oldest house built in {}. Newest house built in {}.'.format(housingdf['Year Built'].min(), housingdf['Year Built'].max()))
housingdf['Year Built'].hist(bins=14, rwidth=.9, figsize=(12,4))
plt.title('When were the houses built?')
plt.show()
# housingdf['Year Built']


# In[21]:


# Let's start with our dependent variable, SalePrice
plt.figure(figsize=(10,6))
sns.distplot(housingdf['SalePrice'])
plt.show()

print('Skewness Score:', (housingdf['SalePrice'].skew()))
print('Kurtosis Score:', (housingdf['SalePrice'].kurtosis()))


#  We can see data is postively skewed therefore there are outliers with this feature majorly towards the right side i.e higher sale prices 

# We will deal with these outliers while doing multivariate analysis. We will do univariate analysis on the data first to look over 

# #### Univariate Analysis for Categorical Variables

# we will draw historgrams to understand the distribution of numerical values

# In[28]:


plt.figure()
housingdf.hist(layout=(8,5),color='k', alpha=0.5, bins=50,figsize=(20,30));


# We can see some numerical features have no variance so they will not be useful for our analysis - 
# MasVNRArea BsmtFinSF2, 2ndFlrSF, LowQualFinSF, BsmtHalfBath, MiscVal, PoolArea, ScreenPorch,3SsnPorch, EnclosedPorch
# 
# we will determine the correlation of these features to our target variable. We will plot a heatmap later to determine the correlation

# #### Univariate Analysis for Categorical Variables

# In[26]:


# Plot bar plot for each categorical feature
fig, axes =plt.subplots(10,4, figsize=(20,40))
axes = axes.flatten()
for ax, catplot in zip(axes, housingdf.dtypes[catcols].index):
    sns.countplot(y=catplot, data=housingdf, ax=ax)

plt.tight_layout()  
plt.show()


# 1. As we can see, some of the categorical variables have no use to us. SO we can remove these
# MSZoning, LotShape, LotConfig, Neighborhood, Condition1, BldgType, Housestyle, Roofstyle, ExterQual, 
# ExterCond, BsmtCond, BsmtFinType1, HeatingQC, CentralAir, KitchenQual, GarageType, GarageQual
# 
# 2. we note that there are plenty of feature were one value is heavily overrpresented, this might be helpful in detecting outliers
# 
# 3. categorical features actually contain rank information in them and should thus be converted to discrete quantitative features. 

# #### Bivariate and multivariate analysis

# In[ ]:




