#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export pandas dataframe to excel
5/15/2021
djkim
"""

import pandas as pd
from openpyxl import load_workbook

d={'chicago':1000,'new york':1300, 'portland':900, 'austin':450,
   'boston':None}

city = pd.Series(d)
city[:2]
city[['new york','austin']]

city2 = pd.DataFrame()
city2['city'] = d.keys()
city2['pop'] = d.values()

# First, create book with one sheet
city2.to_excel('city.xlsx', sheet_name='city', index=False)

# And then append many sheets using openpyxl
book = load_workbook('city.xlsx')
with pd.ExcelWriter('city.xlsx', engine='openpyxl', mode='a') as writer:
    city2.to_excel(writer, sheet_name='city2', index=False, startrow=1)
    city2.to_excel(writer, sheet_name='city3', index=False, startrow=2)
    
# Or create a book with many sheets, without appending
with pd.ExcelWriter('city2.xlsx') as writer:
    city2.to_excel(writer, sheet_name = 'x1')
    city2.to_excel(writer, sheet_name = 'x2', index=False, startrow=1)

