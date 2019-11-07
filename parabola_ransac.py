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
    # see http://en.wikipedia.org/wiki/RANSAC#The_algorithm

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

def distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt((dx * dx) + (dy * dy))

def to_eol(x1, y1, x2, y2):
    # convert line segment to form y = mx + b
    rise = x2 - x1
    run = y2 - y1
    if run == 0:
        # vertical lines have infinite slope
        return None, None
    m = rise / float(run)
    # substitute m into y=mx+b, b = y-mx
    b = y1 - (m * x1)
    return m, b

#
#

def paint(image, features, colour, csize=2):
    for item in features:
        if not len(item):
            continue
        if len(item[0]) == 2:
            x, y = item[0]
            xy = int(x), int(y)
            cv2.circle(image, xy, csize, color=colour, thickness=-1)
        elif len(item[0]) == 4:
            for x1,y1,x2,y2 in item:
                cv2.line(image,(x1,y1),(x2,y2),colour,csize)

#
#

def track(image):

    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    corners = 60
    quality = 0.25
    min_distance = 10

    # Apply edge detection method on the image
    f = cv2.Canny(grey, 50, 20, apertureSize=3)
    #f = cv2.goodFeaturesToTrack(grey, corners, quality, min_distance)

    nonzero = np.nonzero(f)

    list = []

    for index in range(len(nonzero[0])):
        list.append([nonzero[1][index], nonzero[0][index]])
        print([nonzero[1][index], nonzero[0][index]])

    return list

#
#

def canny(im):

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 35, 150, 200)

    min_thresh, max_thresh, apertureSize = 100, 200, 3
    edges = cv2.Canny(gray, min_thresh, max_thresh, apertureSize = apertureSize)

    return edges

def plotParabola(img, model):
    a, b, c, cost = model
    #X = np.array([x[0] for x in inliners])
    X = np.linspace(0, 900)  # draw 100 continuous points directly from 0 to 15
    for x in X:
      y = a * (x ** 2) + b * x + c
      xy = int(x), int(y)
      cv2.circle(img, xy, 2, color=(255,0,0), thickness=2)
    #plt.plot(X, y, color="red", label="solution line", linewidth=2)
    #plt.legend()  # Draw Legend
    #plt.show()

def getBestFit(results):
    return max([d for d in results], key=lambda x: x[2])

p = optparse.OptionParser()
p.add_option("-H", "--hough", dest="hough", action="store_true")
p.add_option("-N", "--nearest", dest="nearest", action="store_true")
p.add_option("-L", "--least-squares", dest="least_squares", action="store_true")
p.add_option("-R", "--ransac", dest="ransac", action="store_true")
p.add_option("-s", "--seek", dest="seek", type="int")
p.add_option("-v", "--video", dest="video")
opts, args = p.parse_args()


frame = 0

# else:
#im = cv2.imread('images/parabola_exemplo1.png')
#im = cv2.imread('images/guitar2.jpg')
im = cv2.imread('images/parabola_exemplo1.png')

#filter_im = cv2.bilateralFilter(im, 35, 150, 200)

edges = canny(im)

filtered = edges
filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

hough_lines = None

draw = im

if 1:
    f = track(filtered)

    n = 3 # min fit
    k = 500 # iterations
    t = 0.2 # threshold
    d = 10 # min number of points within threshold

    model_iter = None

    results = ransac(f, LmsModel(), n, k, t, d)

    bestfit = getBestFit(results)

    plotParabola(draw, bestfit[0])


plt.imshow(draw, interpolation='nearest')
plt.show()