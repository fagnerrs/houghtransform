from HoughParabola import *
import cv2
from matplotlib import pyplot as plt

# Create binary image and call hough_line
image = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0]])

#image = np.zeros((50,50))
#image[10:40, 10:40] = np.eye(30)

hough_parabola(image, [0, 90, 1])

exit()

##--------------------------------------------

img = cv2.imread('images/exemplo1.jpg')
image = cv2.bilateralFilter(img, 35, 150, 200)

# Apply edge detection method on the image
edges = cv2.Canny(image, 50, 20, apertureSize=3)

#ret, thresh1 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV)

accumulator, thetas, rhos = hough_line(edges, [-180, 180, 1])

# Easiest peak finding based on max votes
idx = np.argmax(accumulator)
maxRho = rhos[int(idx / accumulator.shape[1])]
maxTheta = thetas[idx % accumulator.shape[1]]
maxVotes = accumulator[int(idx / accumulator.shape[1]), int(idx % accumulator.shape[1])]

print("MAX: rho={0}, theta={1}, votes={2}".format(maxRho, np.rad2deg(maxTheta), maxVotes))

peeks = np.nonzero(accumulator)
newImg = np.zeros(image.shape)

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

  #print("x={1}, y={0:}".format(int(x), int(y)))

  #m = -(1/np.tan(theta))
  #c = rho * (1/np.sin(theta))

  #ptY = m * x + c

  pt1 = (int(x + 1000 * -a), int(y + 1000 * b))
  pt2 = (int(x - 1000 * -a), int(y - 1000 * b))

  print("ptX={1}, ptY={0:}".format(pt1, pt2))

  cv2.line(img, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
plt.imshow(img, interpolation='nearest')
plt.show()