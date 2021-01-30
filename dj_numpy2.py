#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:02:40 2019

@author: djkim
"""

import numpy as np
a=np.arange(15).reshape(3,5)
a

a.size
a.shape

b=np.array([6,7,8])
b

print(type(a))
print(type(b))

np.linspace(0,1,10)

A=np.array([[1,1],[0,1]])
B=np.array([[2,0],[3,4]])
#elementwise product
A*B
#matrix product
A@B

a=np.random.normal(size=5)
print(a)
a=np.random.randint(10,30,6)
a
a=np.random.random((10,))
a
type(a)
a=np.random.random(10)
a
type(a)
amin, amax = a.min(), a.max()
print(amin, amax)

x=np.arange(10)
np.random.shuffle(x)
print(x)

def roll_dice(num_face):
    return np.random.randint(1, num_face)

print(roll_dice(6))
