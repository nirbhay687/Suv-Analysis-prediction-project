#!/usr/bin/env python
# coding: utf-8

# ### Problem: A car company has released a new SUV in the market. Using the previous data about sales of their SUVs, predict the group of people who might be interested in buying it.

# ### Importing Libraries

# In[4]:


import sklearn
import pandas as pd
import seaborn as sns
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# ### Reading and Summerizing the SUV Dataset

# In[13]:


data_set = pd.read_csv ('C:/Users/HP/Documents/dataset/suv_data.csv')


# In[15]:


data_set.head ()


# In[16]:


data_set.groupby ('Purchased').size()


# In[18]:


cleaned_data_set = data_set.drop (columns = ['User ID'], axis = '1')
cleaned_data_set.head ()


# ### Visualization

# In[20]:


sns.countplot (x = 'Purchased', data = cleaned_data_set)


# In[22]:


sns.countplot ( x = 'Purchased', hue = 'Gender', data = cleaned_data_set)


# In[27]:


binary_gender = pd.get_dummies (cleaned_data_set ['Gender'],drop_first = True)
binary_gender.head ()


# In[28]:


final_data_set = pd.concat([cleaned_data_set,binary_gender], axis=1)
final_data_set = final_data_set.drop(columns = ['Gender'], axis = 1)
final_data_set.head ()


# In[30]:


X = final_data_set.drop(columns=['Purchased'],axis=1)
y = final_data_set['Purchased']


# In[31]:


X.head()


# In[32]:


y.head()


# since values in Age and EstimatedSalary are in wide range, we need to scale it

# In[34]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# Splitting the training and Testing data

# In[37]:


X_train, X_test, y_train, y_test =model_selection.train_test_split (X,y, test_size = 0.3, random_state = 2)


# ### Model Training

# In[39]:


classifier = LogisticRegression (random_state = 2, )
classifier.fit (X_train, y_train)


# In[40]:


y_pred = classifier.predict (X_test)


# In[41]:


accuracy_score (y_test, y_pred)


# Thus the trained model can accurately predict the Group of customers that might buy a SUV, when customer data is fed.

# In[ ]:




