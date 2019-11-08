import numpy as np
from Hough import *
from HoughParabola import *
import cv2
from matplotlib import pyplot as plt

class Hough:

  def find_lines(self, image, edges, rangeTheta):
    accumulator, thetas, rhos =  self.hough_transformation(edges, rangeTheta)
    self.draw_lines(image, edges, accumulator, rhos, thetas)

  def hough_transformation(self, edges, rangeTheta):

    # theta: the angles used in the line function
    thetas = np.deg2rad(np.arange(rangeTheta[0], rangeTheta[1], rangeTheta[2]))

    # get image size
    height, width = edges.shape

    #Max distance of the image: euclidean distance 2d
    diag_len = np.ceil(np.sqrt(width * width + height * height))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

    # cos and sin of the given thetas
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    #matrix [number of rhos, number og thetas]
    zeros = (int(2 * diag_len), int(num_thetas))

    # Hough accumulator array of theta vs rho
    accum = np.zeros(zeros, dtype=float)

    y_idxs, x_idxs = np.nonzero(edges)  #

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
      x = x_idxs[i]
      y = y_idxs[i]

      for t_idx in range(num_thetas):
        # Calculate rho. diag_len is added for a positive index
        rho = round(x * cos_t[t_idx] + y * sin_t[t_idx])

        # Y = valor de P, X = índice theta
        accum[int(rho), int(t_idx)] +=  1

    return accum, thetas, rhos

  def draw_lines(self, image, edges, accumulator, rhos, thetas):

    # Easiest peak finding based on max votes
    idx = np.argmax(accumulator)
    maxRho = rhos[int(idx / accumulator.shape[1])]
    maxTheta = thetas[idx % accumulator.shape[1]]
    maxVotes = accumulator[int(idx / accumulator.shape[1]), int(idx % accumulator.shape[1])]

    print("MAX: rho={0}, theta={1}, votes={2}".format(maxRho, np.rad2deg(maxTheta), maxVotes))

    peeks = np.nonzero(accumulator)

    threshold = 100
    for index in range(len(peeks[0])):

      accumX = peeks[1][index]
      accumY = peeks[0][index]

      theta = thetas[accumX]
      rho = accumY

      if accumulator[accumY, accumX] <= (maxVotes - threshold):
        continue

      print("rho={0}, theta={1}, votes={2}".format(rho, np.rad2deg(theta), accumulator[accumY, accumX]))

      x = rho * np.cos(theta)
      y = rho + np.sin(theta)

      a = np.sin(theta)
      b = np.cos(theta)


      pt1 = (int(x + 1000 * -a), int(y + 1000 * b))
      pt2 = (int(x - 1000 * -a), int(y - 1000 * b))

      print("ptX={1}, ptY={0:}".format(pt1, pt2))

      cv2.line(image, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
      cv2.line(edges, pt1, pt2, (0, 0, 0), 15, cv2.LINE_AA)

    #plt.imshow(edges, interpolation='nearest')
    plt.imshow(image, interpolation='nearest')
    plt.show()

    return image, edges