#!/usr/bin/env python
# coding: utf-8

# # Collecting Data: Importing Libraries
# 

# In[27]:


import pandas as pd
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import pickle

heart_diseases_dataset = pd.read_csv(r"C:\Users\Syed\Desktop\Studies\7th Semester\Data Mining\Project\Data\datasets.csv")
heart_diseases_dataset.head(15)


# # Data Visualization

# In[28]:


# Sex vs Target (male = 1 , female = 0)
sns.countplot(x= "target", hue= "sex", data= heart_diseases_dataset)


# In[29]:


#Target vs Fasting blood suger (fbs) ( true = 1 , false = 0)
sns.countplot(x= "target", hue= "fbs", data= heart_diseases_dataset)


# In[30]:


#Target vs chest pain (cp)
sns.countplot(x= "target", hue= "cp", data= heart_diseases_dataset)


# In[31]:


#Target vs ECG
sns.countplot(x= "target", hue= "restecg", data= heart_diseases_dataset)


# In[32]:


#Plotting age histogram
heart_diseases_dataset['age'].plot.hist()


# # Data Wrangling

# In[33]:


heart_diseases_dataset.isnull().sum()


# In[34]:


sp = pd.get_dummies(heart_diseases_dataset['slope'], prefix = 'slope')
th = pd.get_dummies(heart_diseases_dataset['thal'], prefix = 'thal')
cp = pd.get_dummies(heart_diseases_dataset['cp'], prefix = 'cp', drop_first = True)
ca = pd.get_dummies(heart_diseases_dataset['ca'], prefix = 'ca', drop_first = True)

heart_diseases_dataset.drop(['cp', 'thal', 'slope', 'ca'], axis =1, inplace = True)
collection = [heart_diseases_dataset, sp, th, cp, ca]
heart_diseases_dataset = pd.concat(collection, axis= 1)


# # Preprocessing

# In[35]:


x = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
hdd = heart_diseases_dataset
hdd[columns_to_scale] = x.fit_transform(hdd[columns_to_scale]) 
hdd.isnull().sum()


# #  Train and Test Data

# In[36]:


#Features
X = hdd.drop('target', axis= 1)
y = hdd['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# # Creating Model

# In[37]:


lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[38]:


predict = lr.predict(X_test)


# In[49]:


confusion_matrix(y_test, predict)


# In[50]:


print("Logistic Regression Accuracy:", accuracy_score(y_test, predict) * 100)


# In[41]:


nb = GaussianNB()
nb.fit(X_train, y_train)


# In[42]:


pred = nb.predict(X_test)
confusion_matrix(y_test, pred)


# In[48]:


print("Naive Bayes Accuracy: ", accuracy_score(y_test, pred) * 100)


# # Saving Model using Pickle

# In[44]:


filename = r'C:\Users\Syed\Desktop\Studies\Python\Heart Disease Prediction\model.sav'
pickle.dump(lr, open(filename, 'wb'))


# In[45]:


# load = pickle.load(open(filename, 'rb'))
# result = load.score(X_test, y_test)
# r = load.predict(x_sample)
# print(result)
# print(r)


# In[47]:


plt.figure(figsize=(20,10))
plt.subplot(2,4,1)
plt.title("LogisticRegression")
sns.heatmap(lr_cmatrix,annot=True,cmap="Blues",fmt="d",cbar=False)


# In[ ]:




