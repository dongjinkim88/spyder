#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:40:21 2020

@author: djkim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n=365
dates = pd.date_range("1/1/2020", periods=n, freq="D")
#dates = pd.date_range("1/1/2020", "12/31/2020")
data = np.random.randn(n)
series = pd.Series(data, dates)

series_w = series.resample("W").mean()

series.plot(label='daily')
series_w.plot(label='weekly mean')
plt.legend()


series_w = series.resample("W").interpolate(method="cubic")
series_w.head(5)
series_w.plot(kind='bar')


months = pd.period_range("2020", periods=12, freq="M")
last_days_1 = months.asfreq("D")+1
last_bdays = last_days_1.to_timestamp() - pd.tseries.offsets.BDay()
last_bdays

# dataframe
people_dict = {
    "weight": pd.Series([68, 83, 112], index=["alice", "bob", "charles"]),
    "birthyear": pd.Series([1984, 1985, 1992], index=["bob", "alice", "charles"], name="year"),
    "children": pd.Series([0, 3], index=["charles", "bob"]),
    "hobby": pd.Series(["Biking", "Dancing"], index=["alice", "bob"]),
}
people = pd.DataFrame(people_dict)

# add remove columns
people["age"] = 2020 - people["birthyear"]  # adds a new column "age"
del people["children"]

# attributes ------------------------------------------------------------------
people.index
people.columns
people.values
people.dtypes

# method ----------------------------------------------------------------------
# When adding a new column, it is added at the end (on the right) by default. 
# You can also insert a column anywhere else using the insert() method:
people.insert(1, "height", [172, 181, 185])
people.eval("body_mass_index = weight / (height/100) ** 2", inplace=True)
threshold=25
people.query('body_mass_index > @threshold and age > @threshold')
people.query('body_mass_index > 25 and age > 25')
people.sort_index() # default is by index
people.sort_index(axis=1, inplace=True) # by column names
people.sort_values(by="age", inplace=True) # by value
people.head()
people.tail()
people.info()
people.describe()
people.plot()
people.hist()
people.height.apply(np.cumsum)
np.cumsum(people.height)


grades_array = np.array([[8,8,9],[10,9,9],[4, 8, 2], [9, 10, 10]])
grades = pd.DataFrame(grades_array, columns=["sep", "oct", "nov"], index=["alice","bob","charles","darwin"])
grades > 5
(grades > 5).all() # default is by index (all rows)
(grades > 5).any(axis=1) # any columns
grades['dec']=np.nan
grades
grades.dropna(axis=1, how='all', inplace=True)
grades
pd.DataFrame([[7.75, 8.75, 7.50]]*4, index=grades.index, columns=grades.columns)


# groupby
final_grades = grades.copy()
final_grades['grade'] = ['b','a','c','a']
final_grades.groupby('grade').mean()

# pivot_table
more_grades = grades.stack().reset_index()
more_grades.columns = ["name", "month", "grade"]
more_grades["bonus"] = [np.nan, np.nan, np.nan, 0, np.nan, 2, 3, 3, 0, 0, 1, 0]
more_grades
pd.pivot_table(more_grades, index="name", values=["grade","bonus"], aggfunc=np.max)
pd.pivot_table(more_grades, index="name", values="grade", columns="month", margins=True)


# my_df.to_csv("my_df.csv")
# my_df_loaded = pd.read_csv("my_df.csv", index_col=0)
# all_cities = pd.merge(left=city_loc, right=city_pop, on="city", how="outer")
# all_cities = city_loc.merge(city_pop, on="city", how="outer")
# df3 = pd.concat([df,df2], axis=1)

