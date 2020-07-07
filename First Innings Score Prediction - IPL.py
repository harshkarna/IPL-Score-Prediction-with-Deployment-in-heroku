

# ## First Innings Score Prediction

# ### Importing essential libraries

# In[20]:


# Importing essential libraries
import pandas as pd
import pickle

# Loading the dataset
df = pd.read_csv('ipl.csv')


# **Pickle :**Pickle in Python is primarily used in serializing and deserializing a Python object structure. In other words, it's the process of converting a Python object into a byte stream to store it in a file/database, maintain program state across sessions, or transport data over the network

# In[21]:


df.head()


# **--- Data Cleaning ---**

# In[22]:


# --- Data Cleaning ---
# Removing unwanted columns
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
df.drop(labels=columns_to_remove, axis=1, inplace=True)


# In[23]:


df.head()


# In[24]:


df['bat_team'].unique()


# #Keeping only that Teams that are currently playing

# In[25]:


# Keeping only consistent teams
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']


# #Filtering out the data:

# In[26]:


df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]


# In[27]:


# Removing the first 5 overs data in every match
df = df[df['overs']>=5.0]


# In[28]:


df.head()


# In[29]:


print(df['bat_team'].unique())
print(df['bowl_team'].unique())


# In[30]:


df['date'].dtype


# In[31]:


# Converting the column 'date' from string into datetime object
from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))


# In[32]:


# --- Data Preprocessing ---
# Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])


# A one hot encoding allows the representation of categorical data to be more expressive. Many machine learning algorithms cannot work with categorical data directly. The categories must be converted into numbers.

# In[33]:


encoded_df.head()


# In[34]:


encoded_df.columns


# Rearranging the column so that our y label i.e **total** will go at last column so as to split data into train and test set.

# In[35]:


# Rearranging the columns(can also be done with iloc)
encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]


# In[36]:


# Splitting the data into train and test set
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]


# In[37]:


y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values


# In[38]:


# Removing the 'date' column
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)


# In[39]:


# --- Model Building ---
# Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[40]:





# ## Ridge Regression

# In[41]:


## Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# Standardisation is already handled by Ridge Regression.

# In[44]:


ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}#
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)


# In[45]:



print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[46]:


prediction=ridge_regressor.predict(X_test)


# In[47]:


import seaborn as sns
sns.distplot(y_test-prediction)


# In[49]:


from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# ## Lasso Regression

# In[50]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[51]:


lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X_train,y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[52]:


prediction=lasso_regressor.predict(X_test)


# In[53]:


import seaborn as sns
sns.distplot(y_test-prediction)


# In[54]:


from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))



# Creating a pickle file for the  regressor
filename = 'first-innings-score-rr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))
