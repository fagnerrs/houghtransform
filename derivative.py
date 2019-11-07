from scipy.misc import *
import numpy as np
import random

def func(d):
  return np.sqrt((d-0)**2 + ((6 - (d**2) - 3) ** 2))


def func1(d):
  return 6 * d ** 3 - 9 * d + 4



def func4(d):
  return 6 * d ** 3 - 9 * d + 4

def func2(d, px, py , a, b, c):
  return np.sqrt((d - px) ** 2 + (a * (px**2) + b * px + c - py) ** 2)

print(derivative(func2, 1, dx=1e-6, args=(0, 0, 1, 1, 0)))
print(derivative(func1, 1, dx=1e-6))

#data = [[1,1],[2, 2], [4, 4,] [5,5],[6,6]]
#print(random.sample(data, 3))


arr = np.array([[1,2,3],
                [4,5,6]])

print(arr.T)
