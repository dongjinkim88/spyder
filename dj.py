# shift+option+F = auto formatting

import sys

print(sys.version)
print(sys.executable)


import requests

r = requests.get("https://www.yahoo.com")
print(r.status_code)

# name = input("your name?")
# print("Hello, ", name)

# !echo $PATH

# print integer and float value 
print("Geeks : % 2d, Portal : % 5.2f" %(1, 05.333))  
print("Geeks : {:2d}, Portal : {:5.2f}".format(1, 05.333))  
print(f'Geeks : {1:2d}, Portal : {05.333:5.2f}')
#f=formatted string


#str.ljust(), str.rjust(), str.centre()
ex = "I love geeksforgeeks"
print(ex.center(40))
print(ex.center(40, '-'))  
print(ex.ljust(40, '-')) 
print(ex.rjust(40, '-')) 

import datetime as dt
t=dt.datetime.now()
print(t.strftime('%A'))
print(t.strftime('%c'))
print(t.strftime('%x'))

'''
np.array, np.arange(start, end, step), np.linspace(start, end, number)
A*B #elementwise product
A@B #matrix product
'''

'''
import np.random as rp
rp.seed(1)
rp.rand(10)
rp.normal(10)
rp.standard_normal(10)

'''

import numpy.random as rp
from scipy.stats import norm, t

rp.seed(1)

#normal sample
z = rp.normal(size=100)

#find the probability (CDF)
p = norm.cdf(z)

#find the probability (percent point function)
t = t.ppf(p, df=4)

print('mu= {:5.4f}, std={:5.4f}'.format(z.mean(),z.std()))
print('mu= {:5.4f}, std={:5.4f}'.format(t.mean(),t.std()))
     
import matplotlib.pyplot as plt
plt.plot(z)
plt.plot(t)

