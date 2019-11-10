from app.models.Image import *
from app.models.ParabolaRansac import *
from app.models.Hough import *

class Parabola:
  def find(self, path, numThreshold, numInliers, numParabolaPoints, numIteractions):

    imageHandler = Image()
    hough = Hough()

    image, edges, nonzeros = imageHandler.getEdges(path)

    angle = hough.findAndClearLines(image, edges, 250)

    edges = imageHandler.rotate(angle, edges, nonzeros)

    image, equation = findParabola(image, edges, numThreshold, numInliers, numParabolaPoints, numIteractions)

    plotAndRotateParabola(image, equation, angle * -1)

    cv2.imwrite(path, image)

    #plt.imshow(image, interpolation='nearest')
    #plt.show()