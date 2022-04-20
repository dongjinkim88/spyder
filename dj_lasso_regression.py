#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lasso regression
Created on Sat Apr 16 11:06:25 2022

@author: djkim
"""

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_diabetes

x,y = load_diabetes(return_X_y=True)
features = load_diabetes()['feature_names']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# using pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso())
    ])

search = GridSearchCV(pipeline, 
                      {'model__alpha':np.arange(0.1,3,0.1)},
                       cv = 5,
                       scoring = 'neg_mean_squared_error',
                       verbose = 3)

search.fit(x_train, y_train)
search.best_params_

coef = search.best_estimator_[1].coef_

print('Features considered by the model:')
print(np.array(features)[coef != 0])

print('Features discarded by the model:')
print(np.array(features)[coef == 0])


# separatly
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

grid = GridSearchCV(Lasso(), 
                      {'alpha':np.arange(0.1,3,0.1)},
                       cv = 5,
                       scoring = 'neg_mean_squared_error',
                       verbose = 3)

grid.fit(x_train_scaled, y_train)
coef = grid.best_estimator_.coef_

print('Features considered by the model:')
print(np.array(features)[coef != 0])

print('Features discarded by the model:')
print(np.array(features)[coef == 0])


# DJ
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

x,y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

x_train_scaled = StandardScaler().fit_transform(x_train)

# to find the optimal alpha
model = LassoCV(cv=5, random_state=0, max_iter=1000)
model.fit(x_train_scaled, y_train)


model = Lasso(alpha=model.alpha_)
model.fit(x_train_scaled, y_train)

coef = model.coef_

print('Features considered by the model:')
print(np.array(features)[coef != 0])

print('Features discarded by the model:')
print(np.array(features)[coef == 0])

print('R squared traing set: ', model.score(x_train_scaled, y_train))
print('R squared test set: ', model.score(StandardScaler().fit_transform(x_test), y_test))
