#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
The purpose of this notebook is Exploratory Data Analysis, 
i.e. analyzing data sets to summarize their main characteristics, often with visual methods.

'''


# In[10]:


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


# In[28]:


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
dataset.head()


# In[20]:


#Statistical distribution
'''
1. There is something wrong with business id: number of unique ids != number of unique business names 
-> some business names have several business_id or business_name is not assigned 
2. Also # of business_names > # of business addresses - several restaurants with same address or some addressses are not listed? 
We have much less inspection scores than inspections - probably won't be a good metrics
3. # of business_adresses > business_location. Location can not be a good metrics, although it would be easier to classify and plot. 
Maybe there is a automatic method how to restore location from the given address
'''
dataset.describe()


# In[31]:


# Transform our 'inspection_date' column to a pandas-understandable format
print(dataset['inspection_date'].head().to_csv(index=False))
dataset['inspection_date'] = pd.to_datetime(dataset['inspection_date'])
print(dataset['inspection_date'].head().to_csv(index=False))


# In[19]:


# Learning how to use groupby 

# applying groupby() function to 
# group the data on business_name value. 
dataset_bn = dataset.groupby('business_name') 

# Let's print the first entries 
# in all the groups formed. 
dataset_bn.first() 


# In[32]:


# Finding the values contained in the "Zzan" group - all inspections for this restaurant
dataset_bn.get_group('Zzan') 


# In[34]:


# Showing the most recent inspection date for each restaurant

dataset_bn_date = dataset.sort_values('inspection_date').groupby(['business_name']).last()
dataset_bn_date


# In[35]:


#Describing the final dataset for the last inspection for each restaurant
dataset_bn_date.describe()


# In[52]:


#Removing all the unnessesary data columns and columns with NaN for inspection-score
dataset_bn_date_cropped = dataset_bn_date.drop(columns = ['business_city', 'business_state', 'business_postal_code', 'business_latitude', 'business_longitude', 'business_phone_number', 'inspection_id', 'violation_id', 'risk_category', 'inspection_type', 'business_address'])
dataset_bn_date_cropped = dataset_bn_date_cropped[dataset_bn_date.inspection_score.notna()]
dataset_bn_date_cropped.head(20)


# In[53]:


#Describing cropped and polished dataset
dataset_bn_date_cropped.describe()


# In[55]:


'''
The address column is not easy applicable to ML algoritm. 
For this attempt I will try to make a prediction on an inspection_score based on existing locations
This is why I need to remove all non-existing locations'''
dataset_bn_date_cropped = dataset_bn_date_cropped[dataset_bn_date_cropped.business_location.notna()]
dataset_bn_date_cropped.describe()

