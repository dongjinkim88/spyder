#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 17:18:41 2021

@author: djkim
"""

from sklearn.linear_model import LinearRegression

import numpy as np
x = 6*np.random.rand(100,1)-3
y = 4+3*x + np.random.randn(100,1)

reg = LinearRegression().fit(x,y)
y_hat = reg.predict(x)

print('intercept = {}, slope = {}, r_square = {}'.format(reg.intercept_[0], reg.coef_[0,0], reg.score(x, y)))

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x,y_hat,'r-')



#practice
model = LinearRegression()
model.fit(x,y)
y_hat = model.predict(x)

print(model.intercept_)
print(model.coef_)
print(model.score(x,y))

plt.figure(figsize=(8,7))
plt.scatter(x,y)
plt.plot(x,y_hat,'r-')
