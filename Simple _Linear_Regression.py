#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Regression series 1 - Simple Linear Regression - sricharan_reddy


# In[2]:


import numpy as np


# In[3]:


import pandas as pd


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


import seaborn as sns


# In[6]:


#we have taken the simple salary dataset which has a linear relation between input and output variables 


# In[7]:


#loading the dataset


# In[8]:


df = pd.read_csv("C:\\Users\\Sricharan Reddy\\Downloads\\Salary_dataset.csv")


# In[9]:


df.head()


# In[10]:


df.head(15)


# In[11]:


df.info()


# In[12]:


#here the column unnamed: 0 is unnecessary we will just drop the column unnamed


# In[16]:


df.drop(columns='Unnamed: 0',inplace=True)


# In[17]:


df.head()


# In[18]:


#lets find out what is the relation between the yearsexperience and salary


# In[20]:


df.describe()


# In[22]:


sns.pairplot(df)
plt.show()


# In[23]:


#we can see the salary and the years of experience go along linearly so we can have a linear model for this problem 


# In[24]:


df.isnull().sum()


# In[25]:


#there are no missing values in this dataset


# In[26]:


df.duplicated().sum()


# In[27]:


#there are no duplicates in the dataset


# In[42]:


x = df[['YearsExperience']]
y = df['Salary']


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2,random_state=9)


# In[ ]:





# In[45]:


from sklearn.linear_model import LinearRegression


# In[46]:


model = LinearRegression()


# In[47]:


model.fit(x_train,y_train)


# In[48]:


model.intercept_


# In[49]:


model.coef_


# In[50]:


#prediction by the model


# In[51]:


train_prediction = model.predict(x_train)


# In[52]:


test_prediction = model.predict(x_test)


# In[53]:


from sklearn.metrics import mean_squared_error


# In[54]:


print("mean squared error of test predictions  : ",mean_squared_error(y_test,test_prediction))


# In[55]:


print("mean squared error of trai prediction : ",mean_squared_error(y_train,train_prediction))


# In[56]:


np.sqrt(6863353.7995688)


# In[57]:


#this is the root mean squared error of our test predictions


# In[58]:


#in any regression model the root mean squared error should be low 


# In[59]:


#we should model using multiple algorithms in regression and should take only which has less root mean squared error 


# In[60]:


#evaluation :


# In[61]:


model.score(x_train,y_train)


# In[62]:


model.score(x_test,y_test)


# In[63]:


#the above metrics are r2 score which tells how much the linear line is better than the average line fit 


# In[65]:


#hence our model looks pretty good with our dataset let's try predicting for new data


# In[67]:


model.predict([[3]])


# In[68]:


#this is predicting that for a person with 3 years for experience would earn around 53990 salary 


# In[ ]:


#therefore this the basic working of linear regression to understand the math behind check out my linkedin post on simple linear regression for more understanding 

