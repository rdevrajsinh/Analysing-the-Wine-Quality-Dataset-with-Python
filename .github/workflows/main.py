#!/usr/bin/env python
# coding: utf-8

# # Analysing the Wine Quality Dataset

# ## Introduction:
# We consider a set of observations on a number of red and white wine varieties involving their chemical properties and ranking by tasters. Wine industry shows a recent growth spurt as social drinking is on the rise. The price of wine depends on a rather abstract concept of wine appreciation by wine tasters, opinion among whom may have a high degree of variability. Pricing of wine depends on such a volatile factor to some extent. Another key factor in wine certification and quality assessment is physicochemical tests which are laboratory-based and takes into account factors like acidity, pH level, the presence of sugar and other chemical properties. For the wine market, it would be of interest if human quality of tasting can be related to the chemical properties of wine so that certification and quality assessment and assurance process is more controlled.
# 
# Two datasets are available of which one dataset is on red wine and have 1599 different varieties and the other is on white wine and have 4898 varieties. Only white wine data is analyzed. All wines are produced in a particular area of Portugal. Data are collected on 12 different properties of the wines one of which is Quality, based on sensory data, and the rest are on chemical properties of the wines including density, acidity, alcohol content etc. All chemical properties of wines are continuous variables. Quality is an ordinal variable with a possible ranking from 1 (worst) to 10 (best). Each variety of wine is tasted by three independent tasters and the final rank assigned is the median rank given by the tasters.
# 
# ![WINE IMAGE](https://img.onmanorama.com/content/dam/mm/en/food/in-season/images/2019/11/8/wine.jpg)

# ## Python Libraries / Modules

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")


# ## Data Content (based on physicochemical tests):
#    - 1  fixed acidity
#    - 2  volatile acidity
#    - 3  citric acid
#    - 4  residual sugar
#    - 5  chlorides
#    - 6  free sulfur dioxide
#    - 7  total sulfur dioxide
#    - 8  density
#    - 9  pH
#    - 10 sulphates
#    - 11 alcohol
#    -    Output variable (based on sensory data): 
#    - 12 quality (score between 0 and 10)

# ## Loading Wine Quality Dataset

# In[6]:


red = pd.read_csv('winequality-red.csv',sep=';')
white = pd.read_csv('winequality-white.csv',sep=';')


# ## Clean and prepare the data.

# In[19]:


white['type']='white'
red['type']='red'


# In[20]:


wine=pd.concat([white,red])


# In[21]:


red.head()


# In[22]:


white.head()


# In[27]:


wine


# In[24]:


print(f'The size of red wine dataset is{red.shape}')


# In[25]:


print(f'The size of white wine dataset is{white.shape}')


# In[26]:


print(f'The size of combined wine dataset is{wine.shape}')


# In[31]:


print("Checking for null values and datatype of column in Red dataset:\n\n")
red.info()


# In[32]:


print("Checking for null values and datatype of column in White dataset:\n\n")
white.info()


# In[33]:


print("Checking for null values and datatype of column in Combined dataset:\n\n")
wine.info()


# In[34]:


print("Checking for null values in Red dataset:\n\n")
red.isnull().sum()


# In[35]:


print("Checking for null values in White dataset:\n\n")
white.isnull().sum()


# In[36]:


print("Checking for null values in combined dataset:\n\n")
wine.isnull().sum()


# ### - Statistical Data Analyze

# In[37]:


print("RED WINE")
red.describe()


# In[38]:


print("White WINE")
white.describe()


# #### From above statistics data we can state that mostly quality of both wine is near to its average quality value

# In[39]:


wine['type'].value_counts()


# ## Data Visualization 
# 

# In[41]:


sns.countplot(x="type", data=wine,palette=['lightgrey','red'])
plt.title("Total count of different types of alcohol")
plt.show()


# #### The above graph shows that white wine has more varieties as compared to red wine in given dataset

# In[47]:


white.hist(bins=25,figsize=(10,10))
plt.show()


# In[68]:


fig2 = plt.figure(figsize=(25,25))
for i, col in enumerate(white):
 if(col!='type' and  col!='quality'):
    ax = plt.subplot(6, 6, i+1)
    ax.set_xlabel(col.capitalize())
    sns.boxplot(x=col, data=white)    
plt.show()


# #### The above plots shows the distribution of White Wine based on different features
# ### Observations regarding variables: All variables have outliers
# 
# - Quality has most values concentrated in the categories 5, 6 and 7. Only a small proportion is in the categories [3, 4] and [8, 9] and none in the categories [1, 2] and 10.
# - Fixed acidity, volatile acidity and citric acid have outliers. If those outliers are eliminated distribution of the variables may be taken to be symmetric.
# - Residual sugar has a positively skewed distribution; even after eliminating the outliers distribution will remain skewed.
# - Some of the variables, e.g . free sulphur dioxide, density, have a few outliers but these are very different from the rest.
# - Mostly outliers are on the larger side.
# - Alcohol has an irregular shaped distribution but it does not have pronounced outliers.

# In[49]:


red.hist(bins=25,figsize=(10,10))
plt.show()


# In[69]:


fig2 = plt.figure(figsize=(25,25))
for i, col in enumerate(red):
 if(col!='type' and  col!='quality'):
    ax = plt.subplot(6, 6, i+1)
    ax.set_xlabel(col.capitalize())
    sns.boxplot(x=col, data=red)    
plt.show()


# #### The above plots shows the distribution of Red Wine based on different features
# ### Observations regarding variables: All variables have outliers
# 
# - Quality has most values concentrated in the categories 5, 6 and 7. Only a small proportion is in the categories [3, 4] and [8, 9] and none in the categories [1, 2] and 10.
# - Fixed acidity, volatile acidity and citric acid have outliers. If those outliers are eliminated distribution of the variables may be taken to be symmetric.
# - Residual sugar has a positively skewed distribution; even after eliminating the outliers distribution will remain skewed.
# - Some of the variables, e.g . free sulphur dioxide, density, have a few outliers but these are very different from the rest.
# - Mostly outliers are on the larger side.
# - Alcohol has an irregular shaped distribution but it does not have pronounced outliers.

# In[53]:


sns.countplot(data=wine,x='quality',hue='type',palette=['lightgrey','red'])
plt.title('Distribution of Wine Quality Scores')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()


# #### From the above plots we can state that quality score for both red and white wine is near to average score

# In[56]:


white=white.drop(['type'],axis=1)
red=red.drop(['type'],axis=1)


# #### Corelation of white wine features is:

# In[57]:


white.corr()


# In[58]:


plt.figure(figsize=[19,10],facecolor='lightyellow')
sns.heatmap(white.corr(),annot=True)


# In[61]:


sns.pairplot(white)


# #### Corelation of red wine features is:

# In[60]:


plt.figure(figsize=[19,10],facecolor='lightgreen')
sns.heatmap(red.corr(),annot=True)


# In[62]:


sns.pairplot(red)


# #### In the above heatmap we can identify relationship between features of wine based on negative or positive value.Negative value represents less relationship.

# ###  Important factors that influence the quality of wine

# In[63]:


# fit a linear regression model to predict quality from all the features
X = white.drop("quality", axis=1) # independent variables
y = white["quality"] # dependent variable
model = LinearRegression() # create a linear regression object
model.fit(X, y) # fit the model to the data

# print the coefficients and intercept of the model
print("Intercept:", model.intercept_)
print("Coefficients:")
for i, col in enumerate(X.columns):
    print(col, ":", model.coef_[i])

# plot a bar chart of the absolute values of the coefficients
plt.bar(X.columns, abs(model.coef_),color='lightyellow')
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Absolute Coefficient")
plt.title("Importance of Features for Wine Quality")
plt.show()


# In[65]:


# fit a linear regression model to predict quality from all the features
X = red.drop("quality", axis=1) # independent variables
y = red["quality"] # dependent variable
model = LinearRegression() # create a linear regression object
model.fit(X, y) # fit the model to the data

# print the coefficients and intercept of the model
print("Intercept:", model.intercept_)
print("Coefficients:")
for i, col in enumerate(X.columns):
    print(col, ":", model.coef_[i])

# plot a bar chart of the absolute values of the coefficients
plt.bar(X.columns, abs(model.coef_),color='cyan')
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Absolute Coefficient")
plt.title("Importance of Features for Wine Quality")
plt.show()


# #### Using Linear Regression we came to conclusion that for both type of wine density can be important feature for wine quality
