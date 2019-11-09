import numpy as np
from Hough import *
from HoughParabola import *
import cv2
from matplotlib import pyplot as plt

class Image:

  def getEdges(self, path):

    img = cv2.imread(path)

    smooth = cv2.bilateralFilter(img, 35, 150, 200)

    edges = cv2.Canny(smooth, 50, 20, apertureSize=3)

    return img, edges