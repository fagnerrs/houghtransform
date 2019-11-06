import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

class ParabolaMls:

  def __func(self, params, x):
   a, b, c = params
   return a*(x**2) + b*x + c

  # Error function, that is, the difference between the value obtained by fitting curve and the actual value
  def __error(self, params, x, y):
   return self.__func(params, x) - y

  # Solving parameters
  def __solveModel(self, X, Y):
    p0 = [10, 10, 10]
    model = leastsq(self.__error, p0, args=(X, Y))
    return model

  def getModel(self, data):

    X = np.array([x[0] for x in data])
    Y = np.array([y[0] for y in data])

    model = self.__solveModel(X, Y)
    a, b, c = model[0]
    cost = model[1]

    return a, b, c, cost
