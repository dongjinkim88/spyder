# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys

print(sys.version)
print(sys.executable)


list = ["apple", "banana", "cherry"]
list.append("orange")
for x in list:
    print(x)
    print(len(x))

list.count("orange")

tuple = ("apple", "banana", "cherry")
for x in tuple:
    print(x)

set = {"apple", "banana", "cherry"}
for x in tuple:
    print(x)
set.add("orange")
print(set)
print(len(set))
print(len(list))
print(len(tuple))

dictionary = {"brand": "Ford", "model": "Mustang", "year": 1964}
print(dictionary)
x = dictionary["model"]

a = 33
b = 33
if b > a:
    print("b is greater than a")
elif a == b:
    print("a and b are equal")
else:
    print("a is greater than b")

i = 0
while i < 6:
    i += 1
    if i == 3:
        continue
    if i == 5:
        break
    print(i)
# print 1 2 4

for i in range(6):  # from 0
    print(i)

for i in range(0, 30, 5):
    print(i)

for x in list:
    for y in set:
        print("{} and {}".format(x, y))

# lambda function can take mult parameters but one expression
f = lambda a, b, c: 3 * a + 2 * b + c
print(f(1, 2, 3))

# lambda is for inside another function
def myfun(n):
    return lambda a: a * n


mydouble = myfun(2)
mytriple = myfun(3)

print(mydouble(10))
print(mytriple(20))


class person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def myfun(self):
        print("hello my name is " + self.name + " and my age is " + str(self.age))


p = person("John", 30)
p.myfun()

iterator = iter(list)
print(next(iterator))
print(next(iterator))

print(list)

import datetime as dt

x = dt.datetime.now()
print(x)

y = dt.datetime(2019, 2, 24)
print(y.strftime("%A"))
print(y.strftime("%c"))
print(y.strftime("%x"))

import re

str = "the rain in spain"

x = re.search("\s", str)  # \s = white space
print(x)

y = re.split("\s", str)
print(y)

x = re.search(r"\bs\w+", str)
print(x.string)


f = open("demo.txt")
print(f.readline())
# print(f.read())
# print(f.read(5))
for x in f:
    print(x)

f = open("demo.txt", "w")
f.write("woops! i have deleted the content!")
f = open("demo.txt", "r")
print(f.read())

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(5, 5), columns=list("abcde"))


def highlight(v):
    if -0.5 < v < 0.5:
        return "background-color: red"
    else:
        return ""


df.style.applymap(highlight, subset=["a", "b"])
with open("highlight.html", "w") as f:
    f.write(
        df.style.applymap(highlight, subset=["a", "b", "c"])
        .set_table_attributes("border=1")
        .render()
    )

