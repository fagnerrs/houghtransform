import numpy as np

def hough_line(img, rangeTheta):
  # theta: the angles used in the line function
  thetas = np.deg2rad(np.arange(rangeTheta[0], rangeTheta[1], rangeTheta[2]))

  # get image size
  height, width = img.shape

  #Max distance of the image: euclidean distance 2d
  diag_len = np.ceil(np.sqrt(width * width + height * height))
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

  # cos and sin of the given thetas
  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)
  num_thetas = len(thetas)

  #matrix [number of rhos, number og thetas]
  zeros = (int(2 * diag_len), int(num_thetas))

  # Hough accumulator array of theta vs rho
  accum = np.zeros(zeros, dtype=float)

  y_idxs, x_idxs = np.nonzero(img)  #

  # Vote in the hough accumulator
  for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]

    for t_idx in range(num_thetas):
      # Calculate rho. diag_len is added for a positive index
      rho = round(x * cos_t[t_idx] + y * sin_t[t_idx])

      # Y = valor de P, X = índice theta
      accum[int(rho), int(t_idx)] +=  1

  return accum, thetas, rhos