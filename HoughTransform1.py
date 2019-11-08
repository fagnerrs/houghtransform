# Python program to illustrate HoughLine
# method for line detection
import cv2
import numpy as np

# Reading the required image in
# which operations are to be done.
# Make sure that the image is in the same
# directory in which this python program is
img = cv2.imread('images/exemplo2.jpg', cv2.IMREAD_GRAYSCALE)

print('Img size', img.shape)

# Apply edge detection method on the image
edges = cv2.Canny(img, 50, 20, apertureSize=3)

# Copy edges to the images that will display the results in BGR
cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# This returns an array of r and theta values
lines = cv2.HoughLines(edges, 1, 360, 150)

# The below for loop runs till r and theta values
# are in the range of the 2d array

# Draw the linesRR
if lines is not None:
  for i in range(0, len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    cv2.line(cdst, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

cv2.namedWindow( "Display window", cv2.WINDOW_NORMAL );
cv2.imshow("Display window", cdst)
cv2.waitKey(0);

