#!/usr/bin/python

import sys
import time
import random
from random import randint
import optparse
import math
from parabola_mls import *
from scipy.misc import *
from matplotlib import pyplot as plt

import cv2

parabolaMls = ParabolaMls()

class LmsModel:

    def __init__(self):
        pass

    def model(self, data):
        return parabolaMls.getModel(data)

    def fit(self, x, y, model):
        a, b, c, cost = model
        d = np.absolute(a * (x ** 2) + b * x + c - y)
        return d

def ransac(data, model, minFit, iteractions, threshold, minpoints):
    results = []
    def iterator():
        for i in range(iteractions):
            sample = random.sample(data, minFit)
            maybe_model = model.model(sample)
            yield maybe_model

    # keep a list of found models
    models = {}
    for maybe_model in iterator():

        inliners = []

        for xy in data:
            f = model.fit(xy[0], xy[1], maybe_model)

            if f < threshold:
                inliners.append(xy)

        if len(inliners) > minpoints:

            this_model = model.model(inliners)

            results.append((this_model, inliners, len(inliners)))


    return results

def plotParabola(img, model):
    a, b, c, cost = model
    X = np.linspace(0, 900, dtype=int)  # draw 100 continuous points directly from 0 to 15

    #for x in X:
    #  y = a * (x ** 2) + b * x + c
    #  xy = int(x), int(y)
    #  cv2.circle(img, xy, 2, color=(255,0,0), thickness=2)

    for i in range(len(X) - 1):
      y1 = a * (X[i] ** 2) + b * X[i] + c
      y2 = a * (X[i + 1] ** 2) + b * X[i + 1] + c

      x1 = X[i]
      x2 = X[i + 1]

      cv2.line(img, (x1, int(y1)), (x2, int(y2)), color=(33, 231, 29), thickness=2)


def getBestFit(results):
    return max([d for d in results], key=lambda x: x[2])


def findParabola(image, edges):
    nonzero = np.nonzero(edges)

    data = []

    for index in range(len(nonzero[0])):
        data.append([nonzero[1][index], nonzero[0][index]])

    numFit = 5 # Number of points to predict parabola
    iteractions = 500 # iterations
    thershold = 0.1 # threshold
    minInliers = 10 # min number of points within threshold

    results = ransac(data, LmsModel(), numFit, iteractions, thershold, minInliers)

    bestfit = getBestFit(results)

    plotParabola(image, bestfit[0])