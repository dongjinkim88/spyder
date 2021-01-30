#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:40:21 2020

@author: djkim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

n=300
df = pd.DataFrame(np.random.randn(n,3), index=pd.date_range('1/1/2019',periods=n), 
                  columns=list('abc'))

window = 10
# moving average
mv_10 = df.rolling(window).mean()

# ewma with half-year half-life
alpha = 2**(-3)   # This is ewma's decay factor.
weight = list(reversed([(1-alpha)**n for n in range(window)]))
ewma = partial(np.average, weights=weight)
mv_ewma = df.rolling(window).apply(ewma)

# plot
plt.figure(1, figsize=(10,15))
plt.style.use('classic')
plt.plot(df['a'], 'b:')
plt.plot(mv_10['a'], 'r-')
plt.plot(mv_ewma['a'], 'k-')
