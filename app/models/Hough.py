from HoughParabola import *
import cv2

class Hough:

  def findAndClearLines(self, image, edges, threshold):
    angles = self.houghTransformation(image, edges, threshold)
    return self.getAngle(angles)

  def houghTransformation(self, image, edges, threshold):

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold, None, 0, 0)

    angles = []

    for i in range(0, len(lines)):
      rho = lines[i][0][0]
      theta = lines[i][0][1]

      angles.append(np.rad2deg(theta))

      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a * rho
      y0 = b * rho
      pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
      pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

      cv2.line(image, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)
      cv2.line(edges, pt1, pt2, (0, 0, 0), 15, cv2.LINE_AA)

    return angles

  def getAngle(self, angles):
    angle = np.absolute(90 - angles[0])
    return angle