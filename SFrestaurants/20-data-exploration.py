#!/usr/bin/env python
# coding: utf-8

# In[8]:


'''
The purpose of this notebook is Exploratory Data Analysis, 
i.e. analyzing data sets to summarize their main characteristics, often with visual methods.

'''


# In[1]:


# Load libraries

from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# Load dataset
filename = "Restaurant_Scores_-_LIVES_Standard.csv"
names = ['business_id', 'business_name', 'business_address', 'business_city', 
         'business_state', 'business_postal_code', 'business_latitude', 'business_longitude', 
         'business_location', 'business_phone_number', 'inspection_id', 'inspection_date', 
         'inspection_score', 'inspection_type', 'violation_id', 'violation_description', 
         'risk_category'
        ]
dataset = read_csv(filename, names=names, low_memory = False)[1:] #removing the first row with names

#Take a first look at the data
print(dataset.head())


# In[3]:


#Statistical distribution
print(dataset.describe())
'''
There is something wrong with business id: number of unique ids != number of unique business names
Also # of business_names > # of business addresses - several restaurants with same address or some addressses are not listed? 
We have much less inspection scores than inspections - probably won't be a good metrics
'''


# In[46]:


# Transform our 'inspection_date' column to a pandas-understandable format
print(dataset.head())
dataset['inspection_date'] = pd.to_datetime(dataset['inspection_date'])
print(dataset.head())


# In[20]:


#Doesn't work yet
##I want to see dates of the inspection on a graph
'''
There is no real time, just date
'''
print(dataset.head())
dataset['inspection_date'] = pd.to_datetime(dataset['inspection_date'])
print(dataset.head())
# print(inspection_date.shape())
# inspection_count = dataset.groupby('inspection_date').size()

# # inspection_count.reshape(1,-1)
# print(inspection_count)
fig, ax = plt.subplots(figsize=(15,10))
ax.set_xticklabels(inspection_date, rotation=90)
plt.hist(inspection_date.sort_values())


# In[45]:


#Doesn't work yet
#Plot data
fig, ax = plt.subplots(figsize=(15,10))
inspection_count.plot()
ax.set_xticklabels(inspection_date, rotation=90)
plt.show()


# In[54]:


# Learning how to use groupby 

# applying groupby() function to 
# group the data on business_name value. 
dataset_bn = dataset.groupby('business_name') 

# Let's print the first entries 
# in all the groups formed. 
dataset_bn.first() 


# In[11]:


# Finding the values contained in the "Zzan" group - all inspections for this restaurant
dataset_bn.get_group('Zzan') 


# In[53]:


# Showing the most recent inspection date for each restaurant

dataset_bn_date = dataset.sort_values('inspection_date').groupby(['business_name']).last()
print(dataset_bn_date)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




