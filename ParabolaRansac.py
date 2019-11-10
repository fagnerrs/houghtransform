import random
from parabola_mls import *
import cv2

parabolaMls = ParabolaMls()

class ParabolaModel:

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
            maybeModel = model.model(sample)
            yield maybeModel

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

def plotAndRotateParabola(img, model, theta):
    a, b, c, cost = model
    X = np.linspace(0, 900, dtype=int)  # draw 100 continuous points directly from 0 to 15

    height, width, color = img.shape

    ox = int(width / 2)
    oy = int(height / 2)

    theta = np.deg2rad(theta)
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)

    for i in range(len(X) - 1):
      y1 = a * (X[i] ** 2) + b * X[i] + c
      y2 = a * (X[i + 1] ** 2) + b * X[i + 1] + c

      x1 = X[i]
      x2 = X[i + 1]

      qx1 = int(ox + cosTheta * (x1 - ox) + -sinTheta * (y1 - oy))
      qy1 = int(oy + sinTheta * (x1 - ox) + cosTheta * (y1 - oy))

      qx2 = int(ox + cosTheta * (x2 - ox) + -sinTheta * (y2 - oy))
      qy2 = int(oy + sinTheta * (x2 - ox) + cosTheta * (y2 - oy))

      cv2.line(img, (qx1, qy1), (qx2, qy2), color=(33, 231, 29), thickness=2)


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

    results = ransac(data, ParabolaModel(), numFit, iteractions, thershold, minInliers)

    bestfit = getBestFit(results)

    return image, bestfit[0]