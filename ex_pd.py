#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


s=pd.Series(np.random.randn(10))
s


# In[6]:


dates=pd.date_range('20190301',periods=6)
dates


# In[7]:


df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('abcd'))
df


# In[8]:


A=df.values
A


# In[16]:


df=pd.DataFrame(A,columns=['a','b','c','d'])
df


# In[17]:


df.dtypes


# In[23]:


df.values


# In[32]:


mask=df.a>0
df.loc[mask,:]


# In[33]:


df.iloc[1:3,2:4]


# In[34]:


df.mean()


# In[35]:


df.apply(lambda x:x.max()-x.min())


# In[36]:


np.cumsum(df)


# In[37]:


df


# In[51]:


df2=pd.DataFrame(np.random.rand(6,2),columns=list('ef'))


# In[53]:


df3=pd.concat([df,df2],axis=1)
df3


# In[54]:


df4=pd.DataFrame(np.random.rand(2,4),columns=list('abcd'))
df4


# In[58]:


dff=pd.concat([df,df4],ignore_index=True)
dff


# In[62]:


df['i']=list('aabccc')


# In[63]:


df.groupby('i').sum()


# In[64]:


pd.read_excel('city.xlsx')


# In[65]:


df.to_csv('test.csv',index=False)


# In[ ]:




