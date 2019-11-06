#!/usr/bin/env python
# coding:utf-8


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


# Data to be fitted
X = np.array([1,2,3,4,5,6])
Y = np.array([9.1,18.3,32,47,69.5,94.8])


# Standard Form of Quadratic Function
def func(params, x):
 a, b, c = params
 return a*(x**2) + b*x + c
 #return a * x * x + b * x + c


# Error function, that is, the difference between the value obtained by fitting curve and the actual value
def error(params, x, y):
 return func(params, x) - y


# Solving parameters
def slovePara():
 p0 = [10, 10, 10]

 Para = leastsq(error, p0, args=(X, Y))
 return Para


# Output the final result
def solution():


  Para = slovePara()
  a, b, c = Para[0]

  print("a=",a," b=",b," c=",c)
  print("cost:" + str(Para[1]))
  print("The curve of solution is:")
  print("y="+str(round(a,2))+"x*x+"+str(round(b,2))+"x+"+str(c))

  plt.figure(figsize=(8,6))
  plt.scatter(X, Y, color="green", label="sample data", linewidth=2)

  # Drawing Fitted Lines
  x = np.linspace(-10, 10) # draw 100 continuous points directly from 0 to 15
  y = a * (x**2) + b * x + c # # function
  plt.plot(x, y, color="red",label="solution line",linewidth=2)
  plt.legend ()# Draw Legend
  plt.show()


solution()