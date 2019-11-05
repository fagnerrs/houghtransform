#!/usr/bin/python

import sys
import time
import random
import optparse
import math

import cv2
import numpy as np

def ls_fit(points):

    if len(points) < 2:
        return None, None

    # see http://hotmath.com/hotmath_help/topics/line-of-best-fit.html
    sum_x2, sum_xy, sum_x, sum_y = 0.0, 0.0, 0.0, 0.0
    for xy in points:
        x, y = xy
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x

    m = sum_xy - ((sum_x * sum_y) / len(points))
    m /= sum_x2 - ((sum_x * sum_x) / len(points))
    c = (sum_y / len(points)) - (m * (sum_x / len(points)))
    return m, c


def ransac_polyfit(points, order=3, n=3, k=5, t=0.1, d=100, f=0.8):
    # Thanks https://en.wikipedia.org/wiki/Random_sample_consensus

    # n – minimum number of data points required to fit the model
    # k – maximum number of iterations allowed in the algorithm
    # t – threshold value to determine when a data point fits a model
    # d – number of close data points required to assert that a model fits well to data
    # f – fraction of close data points required

    x = [x[0] for x in points]
    y = [y[0] for y in points]


    bestfit = np.polyfit(x, y, order)

    return bestfit

#
#   LMS model used for RANSAC implementation

class LmsModel:

    def __init__(self):
        pass

    def model(self, data):
        # calculate a model from a given dataset
        print(ransac_polyfit(data))

        return ls_fit(data)

    def fit(self, x, y, model):
        # Distance from an x,y point to a given model
        # see http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        # model in y=mx+c form, so
        # ax + by + c = 0, a==m, b==-1, c==c
        m, c = model
        d = math.fabs((m * x) - y + c) / math.sqrt((m * m) + 1)
        return d

    def error(self, model, data):
        # Average RMS error in fit of data to model
        sum = 0
        for x, y in data:
            f = self.fit(x, y, model)
            sum += f * f
        # divide by the number of points,
        # otherwise good matches with more points look worse
        return math.sqrt(sum) / len(data)

#
#

def ransac(data, model, n, k, t, d, model_iter=None, all_results=False):
    # see http://en.wikipedia.org/wiki/RANSAC#The_algorithm

    data = [tuple(x[0]) for x in data]

    if len(data) < n:
        if all_results:
            return []
        return None, None, None

    best_model = None
    best_consensus_set = None
    best_error = None

    results = []

    if model_iter is None:
        def iterator():
            # Standard RANSAC iterator
            for i in range(k):
                sample = random.sample(data, n)
                maybe_model = model.model(sample)
                yield maybe_model
    else:
        iterator = model_iter

    # keep a list of found models
    models = {}

    for maybe_model in iterator():

        consensus_set = []

        for xy in data:
            f = model.fit(xy[0], xy[1], maybe_model)
            if f < t:
                consensus_set.append(xy)

        if len(consensus_set) > d:
            this_model = model.model(consensus_set)

            if models.get(this_model):
                continue

            models[this_model] = True

            this_error = model.error(this_model, consensus_set)

            results.append((this_error, this_model, consensus_set))

            if (best_error is None) or (this_error < best_error):
                best_model = this_model
                best_error = this_error
                best_consensus_set = consensus_set

    results.sort()

    if all_results:
        return results

    return best_model, best_consensus_set, best_error

#
#

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
    #f = cv2.Canny(grey, 50, 20, apertureSize=3)
    f = cv2.goodFeaturesToTrack(grey, corners, quality, min_distance)

    try:
      criteria = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01
      cv2.cornerSubPix(grey, f, (5,5), (-1,-1), criteria)
    except:
      print('erro 1')
      return []

    return f

#
#

def canny(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    min_thresh, max_thresh, apertureSize = 100, 200, 3
    edges = cv2.Canny(gray, min_thresh, max_thresh, apertureSize = apertureSize)

    return edges

#
#

def draw_line(draw, m, c, colour):
    x0 = 0
    x1 = draw.shape[1]
    y0 = c
    y1 = (m * x1) + c
    cv2.line(draw, (0, int(y0)), (int(x1), int(y1)), color=colour)

def overlap(d, l):
    for item in l:
        if d.get(item):
            return True
    return False

def show_models(draw, results, colour):
    done = {}
    print("*" * 20)
    for error, model, points in results:
        if overlap(done, points):
            continue
        m, c = model
        draw_line(draw, m, c, colour)

        for point in points:
            done[point] = True

        #points = [ [(x,y,)] for x,y in points ]
        #paint(draw, points, colour)
        print(error, model)
        points.sort()
        points.reverse()
        start = points[0]
        for point in points[1:]:
            d = distance(start[0], start[1], point[0], point[1])
            print(d)

#
#

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
im = cv2.imread('images/exemplo1_th1.jpg')

#filter_im = cv2.bilateralFilter(im, 35, 150, 200)

edges = canny(im)

filtered = edges
filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

while frame < 24:

    frame += 1
    hough_lines = None

    draw = im

    if 1:
        #f = np.nonzero(im)
        f = track(im)
        colour = 0, 0, 255
        #paint(draw, f, colour, csize=2)

        if (1 == 1):
            n = 3 # min fit
            k = 1000 # iterations
            t = 0.5 # threshold
            d = 3 # min number of points within threshold

            model_iter = None
            if not hough_lines is None:

                def model_iter():
                    for line in hough_lines[0]:
                        x0, y0, x1, y1 = line
                        model = to_eol(x0, y0, x1, y1)
                        if model == (None, None):
                            continue
                        yield model

            results = ransac(f, LmsModel(), n, k, t, d, all_results=True, model_iter=model_iter)

            if len(results):
                colour = 0, 255, 128
                show_models(draw, results, colour)

    image_path = "images/guitar_result.png"

    if frame == 24:
        cv2.imwrite(image_path, draw)
        break



    def getkey():
        return cv2.waitKey(10) & 0xFF

    key = getkey()

    if key == ord(' '):
        while True:
            key = getkey()
            if key != 255:
                break
            time.sleep(0.1)

    if key == 27:
        break
    elif key == ord('v'):
        print("Saving frame as", image_path)
        cv2.imwrite(image_path, draw)

print("Frames", frame)
#cv2.imshow(draw)
#cv2.show()

# FIN