#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 21:45:48 2022

@author: djkim
"""

import numpy as np
import pandas as pd

FTSE100 = pd.read_csv('FTSE100.csv')
df = FTSE100.copy()
df.rename(columns={'FTSE100':'price'}, inplace=True)

# log return
# df['log_rtn'] = np.log(1+df['FTSE100'].pct_change())
df['log_rtn'] = np.log(df.price/df.price.shift(1))
mu = df.loc[1:,'log_rtn'].mean()
df.loc[0,'log_rtn'] = mu

# error square
df['e_square'] = np.square(df.log_rtn - mu)
var = df.loc[1:,'e_square'].mean()
df.loc[0,'e_square'] = var

# sigma square (conditional variance)
df['sigma_square'] = df['e_square']
# for i in range(df.shape[0]-1):
#     df.loc[i+1,'sigma_square'] = omega + alpha*df.loc[i,'e_square'] + beta*df.loc[i,'sigma_square']

# # log likelihood
# df['log_likelihood'] = (-np.log(np.sqrt(df['sigma_square'])) 
#                         - 0.5*df['e_square']/df['sigma_square'])
# df.loc[0,'log_likelihood'] = 0.0
# df.log_likelihood.sum()

# maximize log likelihood = minimize minus log likelihood
import scipy.optimize as opt

# objective function: minimize minus log likelihood
def obj_fun(x):
    df.loc[0,'sigma_square'] = df.loc[0,'e_square']
    for i in range(df.shape[0]-1):
        df.loc[i+1,'sigma_square'] = x[0] + x[1]*df.loc[i,'e_square'] + x[2]*df.loc[i,'sigma_square']
    
    # log likelihood
    df['log_likelihood'] = (-np.log(np.sqrt(df['sigma_square'])) 
                            - 0.5*df['e_square']/df['sigma_square'])
    df.loc[0,'log_likelihood'] = 0.0
    return -df.log_likelihood.sum()

# constraints: alpha + beta <= 1 or 1 - alpha - beta > 0
constraints = ({'type':'ineq','fun': lambda x:1.0-x[1]-x[2]})

# omega, alhpa, beta >= 0
bounds = [(0,None), (0,1), (0,1)]

# initialize
omega = 0.0
alpha = 0.1
beta = 0.9
x = [omega, alpha, beta]

results = opt.minimize(obj_fun, x, method='SLSQP', bounds=bounds, constraints=constraints)

print(results)
print(f'max log likelihood = {-results.fun:.2f}')
print(f'omega = {results.x[0]:.4f}')
print(f'alpha = {results.x[1]:.4f}')
print(f'beta= {results.x[2]:.4f}')

print(obj_fun([0.01,0.1,0.9]))
print(obj_fun([9.9973e-07,0.0869,0.9055]))



import numpy as np
import pandas as pd
from arch import arch_model

FTSE100 = pd.read_csv('FTSE100.csv')

# returns = 100 * FTSE100['FTSE100'].pct_change().dropna()
returns = df.log_rtn.copy()

model = arch_model(returns, p=1, q=1)

model_fit = model.fit()
model_fit.summary()

pred = model_fit.forecast(horizon=1)
np.sqrt(pred.variance.values[-1,:][0])



from random import gauss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# create dataset
n = 1000
omega = 0.5

alpha_1 = 0.1
alpha_2 = 0.2

beta_1 = 0.3
beta_2 = 0.4

test_size = int(n*0.1)

series = [gauss(0,1), gauss(0,1)]
vols = [1, 1]

for _ in range(n):
    new_vol = np.sqrt(omega + alpha_1*series[-1]**2 + alpha_2*series[-2]**2 + beta_1*vols[-1]**2 + beta_2*vols[-2]**2)
    new_val = gauss(0,1) * new_vol
    
    vols.append(new_vol)
    series.append(new_val)
    
plt.figure(figsize=(10,4))
plt.plot(series)
plt.title('Simulated GARCH(2,2) Data', fontsize=20)

plt.figure(figsize=(10,4))
plt.plot(vols)
plt.title('Data Volatility', fontsize=20)

plt.figure(figsize=(10,4))
plt.plot(series)
plt.plot(vols, color='red')
plt.title('Data and Volatility', fontsize=20)

# PACF Plot
plot_pacf(np.array(series)**2)
plt.show()

# Fit the GARCH Model
train, test = series[:-test_size], series[-test_size:]
model = arch_model(train, p=2, q=2)
model_fit = model.fit()
model_fit.summary()

# Predict
predictions = model_fit.forecast(horizon=test_size)

plt.figure(figsize=(10,4))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(np.sqrt(predictions.variance.values[-1, :]))
plt.title('Volatility Prediction', fontsize=20)
plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)

predictions_long_term = model_fit.forecast(horizon=1000)
plt.figure(figsize=(10,4))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(np.sqrt(predictions_long_term.variance.values[-1, :]))
plt.title('Long Term Volatility Prediction', fontsize=20)
plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
    
# Rolling Forecast Origin
rolling_predictions = []
for i in range(test_size):
    train = series[:-(test_size-i)]
    model = arch_model(train, p=2, q=2)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

plt.figure(figsize=(10,4))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(rolling_predictions)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
