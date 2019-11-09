from Image import *
from ParabolaRansac import *

imageHandler = Image()
hough = Hough()


image, edges = imageHandler.getEdges('images/exemplo1.jpg')

hough.find_lines(image, 100, edges, [-180, 180, 1])

findParabola(image, edges)

plt.imshow(image, interpolation='nearest')
plt.show()



