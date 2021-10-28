#!/usr/bin/env python
# coding: utf-8

# # STEP #0: PROBLEM STATEMENT

# - Data represents weekly 2018 retail scan data for National retail volume (units) and price. 
# - Retail scan data comes directly from retailers’ cash registers based on actual retail sales of Hass avocados. 
# - The Average Price (of avocados) in the table reflects a per unit (per avocado) cost, even when multiple units (avocados) are sold in bags. 
# - The Product Lookup codes (PLU’s) in the table are only for Hass avocados. Other varieties of avocados (e.g. greenskins) are not included in this table.
# 
# Some relevant columns in the dataset:
# 
# - Date - The date of the observation
# - AveragePrice - the average price of a single avocado
# - type - conventional or organic
# - year - the year
# - Region - the city or region of the observation
# - Total Volume - Total number of avocados sold
# - 4046 - Total number of avocados with PLU 4046 sold
# - 4225 - Total number of avocados with PLU 4225 sold
# - 4770 - Total number of avocados with PLU 4770 sold
# 
# 

# ![image.png](attachment:image.png)
# Image Source: https://www.flickr.com/photos/30478819@N08/33063122713

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

# In[1]:


# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import random
import seaborn as sns
from fbprophet import Prophet


# In[2]:


# dataframes creation for both training and testing datasets 
avocado_df = pd.read_csv('avocado.csv')


# # STEP #2: EXPLORING THE DATASET  

# In[3]:


# Let's view the head of the training dataset
avocado_df.head()


# In[4]:


# Let's view the last elements in the training dataset
avocado_df.tail(20)


# In[5]:


avocado_df = avocado_df.sort_values("Date")


# In[6]:


plt.figure(figsize=(10,10))
plt.plot(avocado_df['Date'], avocado_df['AveragePrice'])


# In[7]:


avocado_df


# In[8]:


# Bar Chart to indicate the number of regions 
plt.figure(figsize=[25,12])
sns.countplot(x = 'region', data = avocado_df)
plt.xticks(rotation = 45)


# In[9]:


# Bar Chart to indicate the year
plt.figure(figsize=[25,12])
sns.countplot(x = 'year', data = avocado_df)
plt.xticks(rotation = 45)


# In[10]:


avocado_prophet_df = avocado_df[['Date', 'AveragePrice']] 


# In[11]:


avocado_prophet_df


# # STEP 3: MAKE PREDICTIONS

# In[12]:


avocado_prophet_df = avocado_prophet_df.rename(columns={'Date':'ds', 'AveragePrice':'y'})


# In[13]:


avocado_prophet_df


# In[14]:


m = Prophet()
m.fit(avocado_prophet_df)


# In[15]:


# Forcasting into the future
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[16]:


forecast


# In[17]:


figure = m.plot(forecast, xlabel='Date', ylabel='Price')


# In[18]:


figure3 = m.plot_components(forecast)


# # PART 2

# In[19]:


# dataframes creation for both training and testing datasets 
avocado_df = pd.read_csv('avocado.csv')


# In[20]:


avocado_df


# In[21]:


avocado_df_sample = avocado_df[avocado_df['region']=='West']


# In[22]:


avocado_df_sample


# In[23]:


avocado_df_sample


# In[24]:


avocado_df_sample = avocado_df_sample.sort_values("Date")


# In[25]:


avocado_df_sample


# In[26]:


plt.figure(figsize=(10,10))
plt.plot(avocado_df_sample['Date'], avocado_df_sample['AveragePrice'])


# In[27]:


avocado_df_sample = avocado_df_sample.rename(columns={'Date':'ds', 'AveragePrice':'y'})


# In[28]:


m = Prophet()
m.fit(avocado_df_sample)
# Forcasting into the future
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[29]:


figure = m.plot(forecast, xlabel='Date', ylabel='Price')


# In[30]:


figure3 = m.plot_components(forecast)

