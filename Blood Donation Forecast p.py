#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway


# In[4]:


# Load the data into a DataFrame
get_ipython().run_line_magic('cd', '"F:\\subm"')
transfusion_data = pd.read_csv('transfusion.data')


# In[5]:


# Display the first few rows of the DataFrame
print(transfusion_data.head())


# In[6]:


# Display information about the dataset
print(transfusion_data.info())


# In[7]:


# Display basic statistics of the dataset
print(transfusion_data.describe())


# In[8]:


# Check for missing values
print(transfusion_data.isnull().sum())


# In[9]:


# Display unique values in each column
for column in transfusion_data.columns:
    print(f"Unique values in {column}: {transfusion_data[column].unique()}")


# In[10]:


# Rename the target column
transfusion_data.rename(columns={'whether he/she donated blood in March 2007': 'target'}, inplace=True)


# In[11]:


# Display the updated DataFrame
print(transfusion_data.head())


# In[12]:


# Display the distribution of the target variable
print(transfusion_data['target'].value_counts(normalize=True))


# In[13]:


# Split the data into features (X) and target variable (y)
X = transfusion_data.drop('target', axis=1)
y = transfusion_data['target']


# In[15]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


from tpot import TPOTClassifier

# Initialize TPOTClassifier
tpot = TPOTClassifier(verbosity=2, random_state=42, generations=5, population_size=20, n_jobs=-1, config_dict='TPOT sparse')


# In[19]:


# Fit the TPOT model
tpot.fit(X_train, y_train)


# In[20]:


# Display the accuracy score of the best model
print("Accuracy score of the best model:", tpot.score(X_test, y_test))


# In[21]:


# Save the best model
tpot.export('blood_donation_model.py')


# In[22]:


# Display the variance of each feature
print(X_train.var())


# In[23]:


# Identify features with high variance
high_variance_features = X_train.columns[X_train.var() > 1]


# In[24]:


# Log normalize the high variance features
X_train[high_variance_features] = np.log1p(X_train[high_variance_features])
X_test[high_variance_features] = np.log1p(X_test[high_variance_features])


# In[27]:


from sklearn.linear_model import LogisticRegression

# Initialize the logistic regression model
model = LogisticRegression(random_state=42)


# In[28]:


# Train the model
model.fit(X_train, y_train)


# In[29]:


# Display the accuracy of the model on the test set
print("Accuracy of the logistic regression model:", model.score(X_test, y_test))


# In[30]:


# Display the coefficients of the features in the logistic regression model
print("Coefficients of the features:", model.coef_)


# In[31]:


# Conclude the project with insights and recommendations
print("Conclusion: The logistic regression model shows an accuracy of X% on the test set. Further improvements can be made by exploring feature engineering, hyperparameter tuning, and other advanced machine learning models.")


# In[33]:


# Assuming 'model' is the trained logistic regression model

# Split the data into features (X) and target variable (y)
X = transfusion_data.drop('target', axis=1)
y = transfusion_data['target']


# In[34]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set (simulating new data)
predictions = model.predict(X_test)


# In[35]:


# Display or use the predictions as needed
print("Predictions for the testing set:")
print(predictions)

# Evaluate the model's accuracy on the testing set
accuracy = model.score(X_test, y_test)
print(f"Accuracy on the testing set: {accuracy}")


# In[ ]:




