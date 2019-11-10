from Image import *
from ParabolaRansac import *
from Hough import *

imageHandler = Image()
hough = Hough()

image, edges, nonzeros = imageHandler.getEdges('images/exemplo2.jpg')

angle = hough.findAndClearLines(image, edges, 250)

edges = imageHandler.rotate(angle, edges, nonzeros)

image, bestFit = findParabola(image, edges)

plotAndRotateParabola(image, bestFit, angle * -1)

plt.imshow(image, interpolation='nearest')
plt.show()



