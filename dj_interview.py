#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 12:01:24 2022

@author: djkim
"""

import os
import pandas as pd
work_dir = os.getcwd()
control = pd.read_csv(os.path.join(work_dir, 'control.csv'))

#-------------------------------------------------------------
# data structure: list, dict, set, tuple
#-------------------------------------------------------------

# 1. data structure notation
# list=[], dict={}, set={}, tuple=()
# 2. how to union two sets (set1 | set2), how to combine two lists/tuples (list1+list2)
# 3. create a list with all even interger until 100
# even = [x for x in range(0,101,2)]
# 4. list vs np.array
# performance
# 5. dictionary: keys(), values(), items()

# Lists are used to store data of different data types in a sequential manner
my_list = [1, 2, 3]
print(my_list)
my_list.append([555, 12]) #add as a single element
print(my_list)
my_list.extend([234, 'more_example']) #add as different elements
print(my_list)
my_list.insert(1, 'insert_example') #add element i
print(my_list)
my_list + [4,5,6] #same as extend
my_list + [[555,12]] #same as append

my_list[-1] #choose the last element

odd = [x for x in range(100) if x % 2 == 1]
odd2 = [x for x in range(1,100,2)]
odd3 = [2*x+1 for x in range(50)]


"""
A common beginner question is what is the real difference between list and np.array. 
The answer is performance. Numpy data structures perform better in:

Size - Numpy data structures take up less space
Performance - they have a need for speed and are faster than lists
Functionality - SciPy and NumPy have optimized functions such as linear algebra operations built in.
"""

import numpy as np
a = np.array([1,2,3])
a.shape # numpy is column-wise
b = np.array([555,12])
np.concatenate([a,b])

# range(start, stop, step)
range(10)
np.arange(10)

range(1,10,2)
np.arange(1,10,2)

print(np.arange(1, 10, 2))
print(np.linspace(1, 9, 5))  # max included
# numpy.linspace(start, stop, num=50)

A = np.arange(6).reshape(2, 3)
B = A.T
# matrix product
A @ B
A.dot(B)
# elementwise product
A * A

# Dictionaries are used to store keys-values pairs = items
my_dict = {'First': 'Python', 'Second': 'Java', 'Third': 'Ruby'}
print(my_dict.keys()) #get keys
print(my_dict.values()) #get values
print(my_dict.items()) #get key-value pairs
print(my_dict.get('First'))

# Sets are a collection of unordered elements that are unique
my_set = {1, 2, 3}
my_set.add(4) #add element to set
print(my_set)

my_set = {1, 2, 3, 4}
my_set_2 = {3, 4, 5, 6}
print(my_set.union(my_set_2), '----------', my_set | my_set_2)
print(my_set.intersection(my_set_2), '----------', my_set & my_set_2)
print(my_set.difference(my_set_2), '----------', my_set - my_set_2)
print(my_set.symmetric_difference(my_set_2), '----------', my_set ^ my_set_2)
my_set.clear()
print(my_set)

# Tuples are the same as lists are with the exception that 
# the data once entered into the tuple cannot be changed no matter what
my_tuple = (1, 2, 3)
my_tuple = my_tuple + (4, 5, 6) #add elements
print(my_tuple)



#-------------------------------------------------------------
# if, elif, else (and or)
#-------------------------------------------------------------
x=10
if x % 2:
    print(f'{x} is odd')
else:
    print(f'{x} is even')


# ternary operator in Python
print(f'{x} is odd') if x % 2 else print(f'{x} is even')

# ternary operator in other languages
# (x % 2) ? print(f'{x} is odd'): print(f'{x} is even')


#-------------------------------------------------------------
# lambda, def, class
#-------------------------------------------------------------
# 1. implement a function and a class for adding two numbers

def add2(a,b):
    return a+b
print('3+4 in function: {}'.format(add2(3,4)))


class add2():
    def __init__(self,a,b):
        self.a=a
        self.b=b
    def run(self):
        return self.a+self.b
c = add2(3,4)
print('3+4 in class: {}'.format(c.run()))

# 2. define the function in an one line
add2 = lambda a,b:a+b
print('3+4 in lambda: {}'.format(add2(3,4)))



#-------------------------------------------------------------
# pandas
#-------------------------------------------------------------
import pandas as pd

# 1. create a table
dict = {'Name':['Tom', 'Tom', 'Jane', 'Jane','Steve'],
        'Class':['math', 'science', 'math', 'art', 'science'],
        'Grade':[80, 90, 70, 100, 85],
        'Rank':[1,2,3,4,5]
       } 
df = pd.DataFrame(dict)  


# 2. combine two tables
df = pd.concat([df, df], ignore_index=True)

# 3. subsetting (I prefer query to filtering)
df.query("Class=='math' and Grade > 60")
df.query(f"Grade >= {df.Grade.median()}")

from pandasql import sqldf
sqldf("select * from df where Class='math' and Grade>60")


max_grade_name = df.loc[df['Grade'].idxmax(),'Name']
sqldf(f"select distinct Name from df where Grade={df.Grade.max()}")
sqldf("select Class, Name, max(Grade) as Grade from df group by Class")

# stats
df.groupby('Class').max()
df.describe()

df.groupby('Class').agg({'Grade':['max','median','min'], 'Name':'count'})
df.pivot_table(index='Class',aggfunc={'Grade':['max','media  n','min'],'Name':'count'})

# 4. pivot vs pivot_table
# reshaping: normalized table to table with index, columns, values
# pivot does not allow duplicate records (pivot_table is fine because of aggfunc)
# below is error because of duplicate
df_p = df.pivot(index='Name', columns='Class', values='Grade')
df_p

# below is ok
df_p2 = df.pivot_table(index='Name', columns='Class', values='Grade', aggfunc='median', fill_value=0)
# df_p2.fillna(0)

# reshaping: table to normalized table
df_n = df.stack().reset_index()
df_n.columns = ['index', 'columns', 'values']
df_n

# math, science: pass if Grade is at least 80
# art: pass if Grade is at least 90
def final(row):
    if row[1] == 'art':
        return 'pass' if row[2] >= 90 else 'fail'
    else:
        return 'pass' if row[2] >= 80 else 'fail'
    
df['Final'] = df.apply(final, axis=1)
# axis=0/1 same as axis='index'/'columns' for output shape
df.sum(axis=0) #output is one row
df.sum(axis=1) #output is one column

#-------------------------------------------------------------
# Excel
#-------------------------------------------------------------
# Reading
# excel = pd.read_excel('city.xlsx') #read the first tab
xls = pd.ExcelFile('city.xlsx')

# to read all sheets
dfs = {}
for sheet_name in xls.sheet_names:
    dfs[sheet_name] = xls.parse(sheet_name)
    
df1 = dfs['x2']
df1 = xls.parse(0)

df = pd.concat([df, df1])

# create a book with many sheets
# if exists, remove and recreat
with pd.ExcelWriter('city.xlsx') as writer:
    df.to_excel(writer, sheet_name='df', index=False, startrow=0)
    df_n.to_excel(writer, sheet_name='df_n', index=False, startrow=0)
    df.to_excel(writer, sheet_name='x2', index=False, startrow=0)
    
# append many sheets into existing book
with pd.ExcelWriter('city.xlsx', engine='openpyxl', mode='a') as writer:
    df.to_excel(writer, sheet_name='df', index=False, startrow=0)
    df_n.to_excel(writer, sheet_name='df_n', index=False, startrow=0)
    df.to_excel(writer, sheet_name='x2', index=False, startrow=11)


with pd.ExcelWriter('city.xlsx', 'openpyxl', mode='a') as writer:
    df.to_excel(writer, sheet_name='y', index=False)



#-------------------------------------------------------------
# Student's t vs Gaussian distributions
#-------------------------------------------------------------

# 1. one to one?
from scipy.stats import norm, t
n=100
rt = t.rvs(size=n, df=4)
# t -> cdf -> ppf (inverse of cdf) in normal
rn = norm.ppf(t.cdf(rt, df=4))

# 2. create time series
import numpy as np
import pandas as pd
df = pd.DataFrame(np.random.randn(n,3), index=pd.date_range('1/1/2022',periods=n), 
                  columns=list('abc'))
df = pd.DataFrame(np.array([rt,rn]).T, index=pd.date_range('1/1/2022',periods=n), 
                  columns=list('ab'))

# 3. calculate the maximum range
df.max() - df.min()

# calculate the interquartile range
s=df.describe()
s.loc['75%']-s.loc['25%']

# 4. 10-day moving average (window=10)
mv_10 = df.rolling(10).mean()

# 5. plot time series and 10-day moving average together
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(np.random.randn(100), index=pd.date_range('1/1/2022', periods=100), columns=['x'])
df['x10'] = df['x'].rolling(10).mean()
plt.plot(df['x'], '-', df['x10'], 'r:')


fig = plt.figure(1)
plt.plot(df['x'], '-', df['x10'], 'r:')
fig.savefig('time_series.png')

with pd.ExcelWriter('time_series.xlsx') as writer:
    df.to_excel(writer, sheet_name='data', index=False)
    # worksheet = writer.sheets['data']
    worksheet = writer.book.add_worksheet('figure')
    worksheet.insert_image('A1', 'time_series.png')
    


#-------------------------------------------------------------
# Linear Regression
#-------------------------------------------------------------
import numpy as np
x = np.random.randn(100,1)
y = 4+3*x + np.random.randn(100,1)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
y_hat = model.predict(x)

print(model.intercept_)
print(model.coef_)
print(model.score(x,y))

import matplotlib.pyplot as plt
plt.figure(figsize=(8,7))
plt.scatter(x,y)
plt.plot(x,y_hat,'r-')

