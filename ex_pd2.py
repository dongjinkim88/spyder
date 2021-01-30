#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x=np.random.uniform(size=50)
y=x*np.random.normal(0,1,size=50)
fig, ax = plt.subplots(2,2)
ax[0,0].plot(x,y)
ax[0,1].scatter(x,y)
ax[1,0].scatter(x,y)
ax[1,1].scatter(x,y)
plt.tight_layout()


#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


d={'chicago':1000,'new york':1300, 'portland':900, 'austin':450,
   'boston':None}
cities = pd.Series(d)
cities


# In[3]:


type(cities)


# In[4]:


mask = cities < 1000
cities[mask]


# In[7]:


city=pd.DataFrame([d])
city


# In[8]:


city2=pd.DataFrame()
city2['city']=d.keys()
city2['pop']=d.values()
city2


# In[9]:


city2.to_excel('city.xlsx', index=False)


# In[10]:


get_ipython().system('ls -l *.xlsx')


# In[11]:


city=pd.read_excel('city.xlsx','Sheet1')
city


# In[13]:


dates=pd.date_range('20190301',periods=6)
dates


# In[14]:


df=pd.DataFrame(np.random.randn(6,4), index=dates,
               columns=list('abcd'))
df


# In[16]:


df.index


# In[17]:


df.columns


# In[18]:


df.describe()


# In[19]:


df[0:3]


# In[25]:


df[['a','d']]


# In[26]:


df.iloc[0:3, 0:2]


# In[28]:


df.iloc[0,1]


# In[29]:


df.apply(np.cumsum)


# In[30]:


np.cumsum(df)


# In[31]:


np.min(df)


# In[32]:


np.min(df.T)


# In[33]:


df=pd.DataFrame({'a':['foo','bar','foo','bar','foo','bar','foo','foo'],
                 'b':np.random.randn(8)})
df


# In[43]:


g=df.groupby('a').sum()
g


# In[44]:


pd.pivot_table(df, values='b', index='a', columns='a')


# In[46]:


ts=pd.Series(np.random.randn(1000),index=pd.date_range('1/1/2000',periods=1000))
ts=ts.cumsum()
ts.plot()


# In[50]:


df=pd.DataFrame(np.random.randn(1000,4), index=ts.index, columns=list('abcd'))
df=df.cumsum()
#plt.figure()
df.plot()
plt.legend()


# In[49]:


x = np.linspace(0, 2, 100)

plt.plot(x, x, label='linear')
plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')

plt.xlabel('x label')
plt.ylabel('y label')

plt.title("Simple Plot")

plt.legend()

plt.show()


# In[56]:


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
#plt.axis('off')
plt.xticks([])
plt.title('TOP')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.title('BOTTOM')
plt.show()


# In[ ]:




