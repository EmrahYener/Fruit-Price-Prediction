
# # STEP #1: IMPORTING DATA

# - You must install fbprophet package as follows: 
#      pip install fbprophet
#      
# - If you encounter an error, try: 
#     conda install -c conda-forge fbprophet
# 
# - Prophet is open source software released by Facebook’s Core Data Science team.
# 
# - Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. 
# 
# - Prophet works best with time series that have strong seasonal effects and several seasons of historical data. 
# 
# - For more information, please check this out: https://research.fb.com/prophet-forecasting-at-scale/
# https://facebook.github.io/prophet/docs/quick_start.html#python-api
# 

# In[3]:


import pandas as pd                   #for data manipulation using dataframes
import numpy as np                    #for data statistical analysis 
import matplotlib.pyplot as plt       #for data visualisation 
import seaborn as sns                 #for data visualisation 
from fbprophet import Prophet         #for future predictions


# In[4]:


df_avocado=pd.read_csv('avocado.csv')


# In[5]:


df_avocado


# In[ ]:





# # STEP #2: EXPLORING THE DATASET  

# In[8]:


df_avocado.head()


# In[9]:


df_avocado.tail()


# ### Sorting values:

# In[10]:


df_avocado=df_avocado.sort_values('Date')


# In[11]:


df_avocado


# In[ ]:





# #### Let's see if we have null values:

# In[7]:


plt.figure(figsize=(10,10))
sns.heatmap(df_avocado.isnull(), cbar=False, cmap='YlGnBu')


# ### Plot the price vs date:

# In[16]:


plt.figure(figsize=(20,10))
plt.plot(df_avocado['Date'], df_avocado['AveragePrice'])


# ### Visualise the count of each region:

# In[22]:


plt.figure(figsize=(20,10))
sns.countplot(x='region', data=df_avocado)  # We count the elements based on the region


# #### As you see, it is really hard to read X labels, lets rotate them:

# In[28]:


plt.figure(figsize=(20,10))
sns.countplot(x='region', data=df_avocado)  # We count the elements based on the region
plt.xticks(rotation=45)


# In[34]:


sns.countplot(x='year', data=df_avocado)  # We count the elements based on the region


# ### Prepare data for Prophet
# #### We need date and average price

# In[41]:


df_avocado_prophet=df_avocado[['Date','AveragePrice']]


# In[42]:


df_avocado_prophet


# # STEP 3: MAKE PREDICTIONS

# In[50]:


df_avocado_prophet=df_avocado_prophet.rename(columns={'Date':'ds', 'Averageprice':"y"})
#We do this renaming because Prophet uses "ds" for x-axis and "y" for y-axis


# In[51]:


df_avocado_prophet


# In[53]:


model=Prophet()
model.fit(df_avocado_prophet)


# In[55]:


future=model.make_future_dataframe(periods=365)


# In[56]:


forecast=model.predict(future)


# In[57]:


forecast


# In[59]:


figure=model.plot(forecast, xlabel='Date', ylabel='Average Price')


# In[60]:


figure=model.plot_components(forecast)


# In[ ]:





# ## Predict for a specific region:

# In[64]:


df_avocado_west=df_avocado[df_avocado['region']=='West']


# In[65]:


df_avocado_west=df_avocado_west.sort_values('Date')


# In[66]:


df_avocado_west


# In[67]:


plt.plot(df_avocado_west['Date'], df_avocado_west['AveragePrice'])


# In[68]:


df_avocado_west=df_avocado_west.rename(columns={'Date':'ds', 'AveragePrice':'y'})


# In[69]:


df_avocado_west_prophet=df_avocado_west[['ds','y']]


# In[70]:


df_avocado_west_prophet


# In[71]:


model_west=Prophet()


# In[73]:


model_west.fit(df_avocado_west_prophet)


# In[74]:


future_west=model.make_future_dataframe(periods=365)


# In[75]:


predict_west=model_west.predict(future_west)


# In[79]:


figure_west=model_west.plot(predict_west, xlabel='Date', ylabel='Average Price')


# In[80]:


figure_west_trend=model_west.plot_components(predict_west)


# In[ ]:





# ### Thanks!
# 
# If you have any question please feel free to contact with me:
# * github.com/EmrahYener
# * linkedin.com/in/emrah-yener
# * xing.com/profile/emrah_yener
# 
# Sources:
# * https://www.udemy.com/course/deep-learning-machine-learning-practical/
# 
# 

# In[ ]:



