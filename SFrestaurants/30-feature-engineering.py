#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
The purpose of this notebook is feature engineering - to either use the current features 
or generate new features based upon them.  The data should also be tested here to ensure that 
the features generated meet expectations and assuptions about them.

'''


# In[58]:


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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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
dataset.head()


# In[8]:


#Modifying the dataset according to 20-load-data
'''
Interestingly, values for unique locations, dates, scores, descriptions are different from the 20-load-data'''
dataset_bn = dataset.groupby('business_name') 
dataset_bn_date = dataset.sort_values('inspection_date').groupby(['business_name']).last()
dataset_bn_date_cropped = dataset_bn_date.drop(columns = ['business_city', 'business_state', 'business_postal_code', 'business_phone_number', 'inspection_id', 'violation_id', 'risk_category', 'inspection_type', 'business_address'])
dataset_bn_date_cropped = dataset_bn_date_cropped[dataset_bn_date.inspection_score.notna()]
dataset_bn_date_cropped = dataset_bn_date_cropped[dataset_bn_date_cropped.business_latitude.notna()]
dataset_bn_date_cropped = dataset_bn_date_cropped[dataset_bn_date_cropped.business_longitude.notna()]
dataset_bn_date_cropped.describe()


# In[11]:


X = dataset_bn_date_cropped.values[:,1:3]
Y = dataset_bn_date_cropped.values[:,5]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# In[85]:


#Actually looking on the data
print('Latitude')
plt.figure(figsize=(16,12))
plt.scatter(y_test, x_test[:,0], color = 'gray')
# plt.plot(x_test[:,1], y_pred, color = 'red', linewidth = 2)
plt.show()


# In[87]:


print('Longitude')
plt.figure(figsize=(16,12))
plt.scatter(y_test, x_test[:,1], color = 'gray')
# plt.plot(x_test[:,1], y_pred, color = 'red', linewidth = 2)
plt.show()


# In[12]:


# Building models - Logistic regression models do badly on Linear Regression model
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
	cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[29]:


'''Surprizingly low scores'''
models = []
models.append(('LR', LinearRegression()))
models.append(('LAS', Lasso()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    cv_results = cross_val_score(model, x_train, y_train, cv = 5)
    print(cv_results)
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[67]:


# Normalizing data - does not affect Linear regression
scaler = StandardScaler()
print(scaler.fit(x_train))
print(scaler.mean_)
print(scaler.transform(x_train))


# In[88]:


model = LinearRegression()
model.fit(x_train, y_train)
#To retrieve the intercept:
print(model.intercept_)
#For retrieving the slope:
print(model.coef_)
y_pred = model.predict(x_train)
print(model.score(x_train, y_train))
df = pd.DataFrame({'Actual':y_train.flatten(),'Predicted':y_pred.flatten()})
df


# In[37]:


model = Ridge(alpha = 1)
model.fit(x_train, y_train)
print(model.score(x_train, y_train))


# In[53]:


model = Lasso(alpha = 0.1, )
model.fit(x_train, y_train)
model.predict(x_train)
print(model.score(x_train, y_train))


# In[ ]:


'''
None of the models work on the chosen dataset yet
'''

