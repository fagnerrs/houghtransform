from Image import *

imageHandler = Image()
hough = Hough()

image = cv2.imread('images/exemplo1.jpg')

edges = imageHandler.getEdges(image)

hough.find_lines(image, edges, [-180, 180, 1])

