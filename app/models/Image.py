from HoughParabola import *
import cv2

class Image:

  def getEdges(self, path):

    img = cv2.imread(path)

    smooth = cv2.bilateralFilter(img, 35, 150, 200)

    edges = cv2.Canny(smooth, 50, 20, apertureSize=3)

    nonzero = np.nonzero(edges)

    data = []

    for index in range(len(nonzero[0])):
        data.append([nonzero[1][index], nonzero[0][index]])

    return img, edges, data

  def rotate(self, angle, edges, nonzeros):

    height, width = edges.shape
    rotatedImg = np.zeros((height, width))

    theta = np.deg2rad(angle)

    ox = int(width/2)
    oy = int(height/2)

    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)

    for i in range(len(nonzeros)):
      x = nonzeros[i][0]
      y = nonzeros[i][1]

      qx = int(ox + cosTheta * (x - ox) + -sinTheta * (y - oy))
      qy = int(oy + sinTheta * (x - ox) + cosTheta * (y - oy))

      if qy > 0 and qx > 0 and qx < width and qy < height:
        rotatedImg[qy, qx] = edges[y, x]

    return rotatedImg