#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[59]:


# Loading Dataset

ipl_auction_df = pd.read_csv('/Users/arijeetbhadra/Downloads/Machine Learning (Codes and Data Files)/Data/IPL IMB381IPL2013.csv')
ipl_auction_df


# In[19]:


ipl_auction_df.head(5)


# In[20]:


# Finding Summary of dataframe

list(ipl_auction_df)


# In[21]:


ipl_auction_df.head(5).transpose()


# In[22]:


ipl_auction_df.shape


# In[23]:


ipl_auction_df.info()


# In[24]:


# Slicing and indexing of dataframe

ipl_auction_df[0:5]


# In[25]:


ipl_auction_df[-5:]


# In[26]:


ipl_auction_df['PLAYER NAME'][0:5]


# In[27]:


ipl_auction_df[['PLAYER NAME','COUNTRY']][0:5]


# In[28]:


ipl_auction_df[['PLAYER NAME','COUNTRY']][5:10]


# In[29]:


# iloc - Function used to select specific range of row & column

ipl_auction_df.iloc[4:9,1:4]


# In[30]:


# Value Count and Cross tabulations

ipl_auction_df.COUNTRY.value_counts()


# In[31]:


ipl_auction_df.COUNTRY.value_counts(normalize=True)


# In[32]:


ipl_auction_df.COUNTRY.value_counts(normalize=True)*100


# In[33]:


pd.crosstab(ipl_auction_df['AGE'],ipl_auction_df['PLAYING ROLE'])


# In[34]:


# Sorting DataFrame by Column Value

ipl_auction_df[['PLAYER NAME','SOLD PRICE']].sort_values('SOLD PRICE')[0:5]


# In[36]:


ipl_auction_df[['PLAYER NAME','SOLD PRICE']].sort_values('SOLD PRICE',ascending=False)[0:10]


# In[37]:


# Creating New Column

ipl_auction_df['premium']= ipl_auction_df['SOLD PRICE']-ipl_auction_df['BASE PRICE']
ipl_auction_df[['PLAYER NAME','BASE PRICE','SOLD PRICE','premium']][0:10]


# In[38]:


ipl_auction_df[['PLAYER NAME','BASE PRICE','SOLD PRICE','premium']].sort_values('premium',ascending=False)[0:10]


# In[39]:


# Grouping & Aggregating


# In[53]:


ipl_auction_df.groupby('AGE')['SOLD PRICE'].mean()


# In[41]:


soldprice_by_age = ipl_auction_df.groupby('AGE')['SOLD PRICE'].mean().reset_index()
soldprice_by_age


# In[42]:


soldprice_by_age_role = ipl_auction_df.groupby(['AGE','PLAYING ROLE'])['SOLD PRICE'].mean().reset_index()
soldprice_by_age_role


# In[43]:


# joing DataFrames


# In[44]:


soldprice_comparision = soldprice_by_age_role.merge(soldprice_by_age,on='AGE',how='outer')
soldprice_comparision


# In[45]:


# Renaming the columns

soldprice_comparision.rename(columns={'SOLD PRICE_x':'SOLD_PRICE_AGE_ROLE','SOLD PRICE_y':'SOLD_PRICE_AGE'},inplace= True)
soldprice_comparision


# In[46]:


soldprice_comparision['change']=soldprice_comparision.apply(lambda rec:(rec.SOLD_PRICE_AGE_ROLE-rec.SOLD_PRICE_AGE)/rec.SOLD_PRICE_AGE,axis=1)


# In[47]:


soldprice_comparision


# In[54]:


# Filtering records by condition


# In[57]:


ipl_auction_df[ipl_auction_df['SIXERS']>80][['PLAYER NAME','SIXERS']]


# In[60]:


# Removing column or Row from dataset

ipl_auction_df.drop('Sl.NO.',inplace= True,axis=1)
ipl_auction_df


# In[61]:


list(ipl_auction_df)


# In[62]:


ipl_auction_df.columns


# # Drawing Plots

# In[64]:


import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[76]:


# Bar Chart

sn.barplot(x='AGE',y= 'SOLD PRICE',data= soldprice_by_age)


# In[1]:


# sn.barplot(x='AGE',y='SOLD_PRICE_AGE_ROLE',hue='PLAYING ROLE',data='soldprice_comparision')


# In[83]:


# Histogram

plt.hist(ipl_auction_df['SOLD PRICE'])


# In[85]:


plt.hist(ipl_auction_df['SOLD PRICE'],bins=20)


# In[68]:


# Distribution /Density plot
sn.distplot(ipl_auction_df['SOLD PRICE'],bins=10)


# In[102]:


# Box Plot

sn.boxplot(ipl_auction_df['SOLD PRICE'])


# In[105]:


box= plt.boxplot(ipl_auction_df['SOLD PRICE'])


# In[107]:


# caps key is box variable return to min and max values of the distribution

[item.get_ydata()[0] for item in box ['caps']]


# In[109]:


# Whiskers key is box variable return to 25 and 75 quantiles

[item.get_ydata()[0] for item in box ['whiskers'] ]


# In[110]:


# Inter quartile range (IQR) is 700000 - 225000 = 475000


# In[111]:


# Median key in box variable returns the median value of the distribution

[item.get_ydata()[0] for item in box['medians']]


# In[115]:


# Finding outliers based on sold Price

ipl_auction_df[ipl_auction_df['SOLD PRICE']>1350000][['PLAYER NAME','PLAYING ROLE','SOLD PRICE']]


# In[118]:


# Comparing Distributions
sn.distplot(ipl_auction_df[ipl_auction_df['CAPTAINCY EXP']==1]['SOLD PRICE'],color='y',label='CAPTAINCY EXP')


# In[124]:


sn.distplot(ipl_auction_df[ipl_auction_df['CAPTAINCY EXP']==0]['SOLD PRICE'],color='r',label='NO CAPTAINCY EXP')


# In[70]:


sn.distplot(ipl_auction_df[ipl_auction_df['CAPTAINCY EXP']==1]['SOLD PRICE'],color='y',label='CAPTAINCY EXP');
sn.distplot(ipl_auction_df[ipl_auction_df['CAPTAINCY EXP']==0]['SOLD PRICE'],color='r',label='NO CAPTAINCY EXP');
plt.legend();


# In[130]:


sn.boxplot(x='PLAYING ROLE',y='SOLD PRICE',data=ipl_auction_df)


# In[131]:


# Few Observations - 

# Median Sold Price for allrounders and batsmen are higher than bowler and wicketkeepers.
# Allrounders who paid more than 1,35,0000 USD are not considered outliers. Allrounders have relative high variance
# There are outliers in batsmen and wicket keeper category. We have found that MS Dhoni is an outlier in Wicket keeper category.


# In[132]:


# Scatter plot


# In[ ]:




