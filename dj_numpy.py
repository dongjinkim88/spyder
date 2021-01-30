#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:40:21 2020

@author: djkim
"""

import numpy as np

# numpy is point concept
a = np.arange(6).reshape(2, 3)
b = a[:, 1:]
b[1, 0] = 100
a
a[1, (0, 2)]

# attributes
a = np.zeros((3, 4))
a.shape
a.size
a.dtype
a.itemsize
type(a)

b = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])
print(np.arange(0, 1 / 2, 0.1))
print(np.linspace(0, 1 / 2, 6))  # max included

c = np.arange(24)
c.shape = (2, 3, 4)
c
c[1, ...]  #  matrix 2, all rows, all columns.  This is equivalent to c[1, :, :]

d = np.arange(6).reshape(2, 3)
d
e = d.T
e
d @ e
d.dot(e)


import numpy.linalg as la

f = np.array([[1, 2, 3], [5, 7, 11], [21, 29, 31]])
la.inv(f)
la.det(f)
eigenvalues, eigenvectors = la.eig(f)
eigenvalues
eigenvectors

# method
np.eye(3)
# method
d.sum()
d.sum(axis=0)
d.sum(axis=1)

# universal functions (ufunc): elementwise
np.square(d)

print("Original ndarray")
print(d)
for func in (
    np.abs,
    np.sqrt,
    np.exp,
#    np.log,
    np.sign,
    np.ceil,
    np.modf,
    np.isnan,
    np.cos,
):
    print("\n", func.__name__)
    print(func(d))


m = np.array([20, -5, 30, 40])
m < [15, 16, 35, 36]
m[m < 30]

np.random.rand(3, 4)
np.random.randn(3, 4)
np.random.randint(3, 10)

