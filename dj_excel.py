#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:21:32 2020

@author: djkim
"""

import numpy as np
import pandas as pd

d={'chicago':1000,'new york':1300, 'portland':900, 'austin':450,
   'boston':None}

city = pd.Series(d)

city2 = pd.DataFrame()
city2['city'] = d.keys()
city2['pop'] = d.values()

# cannot add sheet, just create a book
city2.to_excel('city2.xlsx', 'data', index=False)

x1 = np.random.randn(100, 2)
df1 = pd.DataFrame(x1)
x2 = np.random.randn(100, 2)
df2 = pd.DataFrame(x2)

# cannot add sheet, just create a book
writer = pd.ExcelWriter('city2.xlsx', engine = 'xlsxwriter')
df1.to_excel(writer, sheet_name = 'x1')
df2.to_excel(writer, sheet_name = 'x2')
writer.save()
writer.close()

# can add sheet 
from openpyxl import load_workbook
book = load_workbook('city2.xlsx')
#book = load_workbook('test.xlsm', keep_vba=True)
writer = pd.ExcelWriter('city2.xlsx', engine='openpyxl')
writer.book = book
city2.to_excel(writer, sheet_name='city2', index=False, startrow=0)
writer.save()
writer.close()
