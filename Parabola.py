from Image import *

imageHandler = Image()
hough = Hough()

image = cv2.imread('images/exemplo2.jpg')

edges = imageHandler.getEdges(image)

hough.find_lines(image, edges, [0, 360, 1])

