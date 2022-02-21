#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 12:01:55 2022

@author: djkim
"""

import pandas as pd
import numpy as np

dict = {'Name':['Tom', 'Tom', 'Jane', 'Jane','Steve'],
        'Class':['math', 'science', 'math', 'art', 'science'],
        'Grade':[80, 90, 70, 100, 85]
       } 
df1 = pd.DataFrame(dict)  

dict2 = {'Name':['Amy', 'Steve'],
        'Class':['art', 'art'],
        'Grade':[90, 80]
       }
df2 = pd.DataFrame(dict2)

df = pd.concat([df1, df2], ignore_index=True)
df.groupby('Class').max()
df.describe()
df.groupby('Class').agg({'Grade':['max','median','min'], 'Name':'count'})

# below are the same
df[['Class','Grade']].groupby('Class').max()
df.groupby('Class').Grade.max()
df.groupby('Class').Grade.agg(['count','mean','std','min','median','max']).T

def qtr1(x):
    return x.quantile(0.25)
def qtr3(x):
    return x.quantile(0.75)
df.groupby('Class').Grade.agg(['count','mean','std','min',qtr1,'median',qtr3,'max']).T


df.query("Class=='math' and Grade > 60")

from pandasql import sqldf
sqldf("select * from df where Class='math' and Grade > 60")

dict3 = {'Class':['math','science','art'],'Test':['Dec','Sep','Oct']}
df3 = pd.DataFrame(dict3)

sqldf("select a.*, b.* from df a join df3 b on b.Class=a.Class")

# math, science: pass if Grade is at least 80
# art: pass if Grade is at least 90

def final(row):
    if row[1] == 'art':
        return 'pass' if row[2] >= 90 else 'fail'
    else:
        return 'pass' if row[2] >= 80 else 'fail'
    
df['Final'] = df.apply(final, axis=1)

# reshaping: table to normalized table
df_n = df.stack().reset_index()
df_n.columns = ['index', 'columns', 'values']
df_n

# reshaping: normalized table to table with index, columns, values
# pivot does not allow duplicate records (pivot_table is fine because of aggfunc)
df_p = df_n.pivot(index='index', columns='columns', values='values')
df_p

# below is error because values are not numeric types to aggregate (default = np.mean())
#df_p2 = df_n.pivot_table(index='index', columns='columns', values='values')
# below is ok
df = pd.concat([df,df], ignore_index=True)
df_p2 = df.pivot_table(index='Name', columns='Class', values='Grade', aggfunc='sum')
df_p2
