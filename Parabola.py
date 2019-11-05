import numpy as np
from Hough import *
from HoughParabola import *
from accumulator import *
import cv2
from matplotlib import pyplot as plt

# Create binary image and call hough_line
#edges = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#                  [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
#                  [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])



#image = np.zeros((50,50))
#image[10:40, 10:40] = np.eye(30)

##--------------------------------------------

img = cv2.imread('images/parabola_exemplo1.png')
image = cv2.bilateralFilter(img, 35, 150, 200)

# Apply edge detection method on the image
edges = cv2.Canny(image, 50, 20, apertureSize=3)

#ret, thresh1 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV)

#nonzeros = np.nonzero(edges)

accumulator, accum2d = hough_parabola(edges, [0, 90, 10])

print(accum2d)

exit()
max = getMaxValue(accumulator)

nonzeros = np.nonzero(accumulator > (max - 10))


#exit()

threshold = 1
for index in range(len(nonzeros[0])):

  y = nonzeros[0][index]
  x = nonzeros[1][index]
  p = nonzeros[2][index]

  #x = accumNonZero[0][x1]
  #y = accumNonZero[1][y1]
  #p = accumNonZero[2][p1]

  print("Y={1}, X={0}, P={2}, count={3}".format(y, x, p, accumulator[y][x][p]))

  plt.plot(x, y, 'r*')

plt.imshow(img, interpolation='nearest')
plt.show()