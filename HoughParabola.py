import numpy as np

def hough_parabola(img, rangeTheta):
  # thetaIndex: the angles used in the line function
  thetas = np.deg2rad(np.arange(rangeTheta[0], rangeTheta[1], rangeTheta[2]))

  # get image size
  height, width = img.shape

  #Max distance of the image: euclidean distance 2d
  diag_len = np.ceil(np.sqrt(width * width + height * height))
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

  # cos and sin of the given thetas
  cosines = np.cos(thetas)
  sines = np.sin(thetas)
  num_thetas = len(thetas)

  #matrix [number of rhos, number og thetas]
  accum = np.zeros((height, width, int(diag_len * 1)))
  accum2d = np.zeros((int(diag_len), int(diag_len)))
  #accum = [[[0 for k in range(height)] for j in range(width)] for i in range(int(diag_len*2))]

  imageY, imageX = np.nonzero(img)

  print('Image Y', imageY)
  print('Image X', imageX)

  diag_len = diag_len * 2

  focusRange = 2

  # Vote in the hough accumulator
  for i in range(len(imageX)):

    x = imageX[i]
    y = imageY[i]

    #for center in range(1, 60):

      #for thetaIndex in range(num_thetas):

        #focusX = center * np.cos(thetas[thetaIndex] * np.pi / 180)
        #focusY = center * np.sin(thetas[thetaIndex] * np.pi / 180)

        #x0 = int(x - focusX)
        #y0 = int(y - focusY)

        #rho = int(center / 1 - sines[thetaIndex])

        #accum[y0][x0][rho] += 1

    #vx = 2
    #vy = 4
    vx = 4
    vy = 3

    #for center in range(focusRange):

      #focus = vy - center
      #directriz = vy + center

      #pf = calculateDistance(x, y, vx, focus)
      #pd = calculateDistance(x, y, x, directriz)

      #if np.absolute(pd - pf) < 150:
        #print('Focus {0}, {1}'.format(vy - focusY, vx, ))
        #print('Directriz {0}, {1}'.format( vy + focusY, x))
        #print('coord Y={0}, X={1}'.format(y, x))
        #print('dist pf={0}, pd={1}'.format(pf, pd))
        #accum[y][x][pd] += 1
        #print('value', accum[y][x][pd])

    for j in range(len(thetas)):
      y0 = y - vy
      x0 = x - vx

      angulo = int(thetas[j])
      angulo = 180

      numerador = (y0 * np.cos(angulo) - x0 * np.sin(angulo)) ** 2
      denominador = 4 * (x0 * np.cos(angulo) + y0 * np.sin(angulo))

      p = 0
      if denominador != 0:
        p = int(numerador / denominador)

        if np.absolute(p) > 0 and abs(p) < int(diag_len * 2) and p != 0:
          print('phi', p)
          print('phi', angulo)
          #accum2d[angulo, p] += 1

  return accum, accum2d

def calculateDistance(x1, y1, x2, y2):
  dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
  return int(dist)


