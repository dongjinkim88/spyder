#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abstract Base Class (ABC) contains one or more abstract methods
Abstract method is a method that is declared, but contains no implementation
Abstract classes cannot be instantiated, and require subclasses to provide
implementations for the abstract methods.

Created on Tue Dec 15 11:32:38 2020
@author: djkim
"""

from abc import ABC, abstractmethod

class ABClass(ABC):
    def __init__(self, value):
        self.value = value
        super().__init__()
        
    @abstractmethod
    def _calculate(self):
        pass
    
class DoAdd42(ABClass):
    def _calculate(self):
        return self.value + 42
    
class DoMul42(ABClass):
    def _calculate(self):
        return self.value * 42
    
x = DoAdd42(10)
y = DoMul42(10)

print(x._calculate())
print(y._calculate())
