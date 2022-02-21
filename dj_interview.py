#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 12:01:24 2022

@author: djkim
"""

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
odd
odd2 = [x for x in range(1,100,2)]
odd2

"""
A common beginner question is what is the real difference between list and np.array. 
The answer is performance. Numpy data structures perform better in:

Size - Numpy data structures take up less space
Performance - they have a need for speed and are faster than lists
Functionality - SciPy and NumPy have optimized functions such as linear algebra operations built in.
"""

import numpy as np
a = np.array([1,2,3])
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
import matplotlib.pyplot as plt
plt.figure(1, figsize=(10,15))
plt.style.use('classic')
plt.plot(df['a'], 'b:')
plt.plot(mv_10['a'], 'r-')



#-------------------------------------------------------------
# pandas
#-------------------------------------------------------------
import pandas as pd

# 1. create a table
dict = {'Name':['Tom', 'Tom', 'Jane', 'Jane','Steve'],
        'Class':['math', 'science', 'math', 'art', 'science'],
        'Grade':[80, 90, 70, 100, 85]
       } 
df = pd.DataFrame(dict)  


# 2. combine two tables
df = pd.concat([df, df], ignore_index=True)

# 3. subsetting (I prefer query to filtering)
df.query("Class=='math' and Grade > 60")

from pandasql import sqldf
sqldf("select * from df where Class='math' and Grade>60")

# stats
df.groupby('Class').max()
df.describe()
df.groupby('Class').agg({'Grade':['max','median','min'], 'Name':'count'})

# 4. pivot vs pivot_table
# reshaping: normalized table to table with index, columns, values
# pivot does not allow duplicate records (pivot_table is fine because of aggfunc)
# below is error because of duplicate
df_p = df.pivot(index='Name', columns='Class', values='Grade')
df_p

# below is ok
df_p2 = df.pivot_table(index='Name', columns='Class', values='Grade', aggfunc='median')
df_p2.fillna(0)

# reshaping: table to normalized table
df_n = df.stack().reset_index()
df_n.columns = ['index', 'columns', 'values']
df_n
