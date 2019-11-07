#!/usr/bin/env python
# coding:utf-8


import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import leastsq


# Data to be fitted
#X = np.array([0, 100, 200])
#Y = np.array([100, 10 ,100])

X = np.array([376, 727, 413, 371, 376, 380, 704, 382, 742, 384])
Y = np.array([260, 269, 522, 219, 375, 400, 340, 352, 176, 390])


M = np.array([[376, 727, 413, 371, 376, 380, 704, 382, 742, 384],
               [260, 269, 522, 219, 375, 400, 340, 352, 176, 390]])

# Standard Form of Quadratic Function
def func(params, x):
 a, b, c = params
 print(x)
 return a*(x**2) + b*x + c

# Error function, that is, the difference between the value obtained by fitting curve and the actual value
def error(params, x, y):
 return func(params, x) - y


# Solving parameters
def slovePara():
 p0 = [1, 1, 1]

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
  X = np.linspace(0, 1000) # draw 100 continuous points directly from 0 to 15



  plt.plot(X, y, color="red",label="solution line",linewidth=2)
  plt.legend ()# Draw Legend
  plt.show()




solution()