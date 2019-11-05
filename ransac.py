import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
from ransacUtils import *
import sys

# Ransac parameters
ransac_iterations = 20  # number of iterations
ransac_threshold = 3  # threshold
ransac_ratio = 0.6  # ratio of inliers required to assert
# that a model fits well to data

# generate sparse input data
# Create binary image and call hough_line
image = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])




n_samples = np.nonzero(image) * 2  # number of input points
outliers_ratio = 0.4  # ratio of outliers

n_inputs = 1
n_outputs = 1

# generate samples
x = 30 * np.random.random((n_samples, n_inputs))

# generate line's slope (called here perfect fit)
perfect_fit = 0.5 * np.random.normal(size=(n_inputs, n_outputs))

# compute output
y = scipy.dot(x, perfect_fit)

#-------------------------------

# add a little gaussian noise
x_noise = x + np.random.normal(size=x.shape)
y_noise = y + np.random.normal(size=y.shape)

# add some outliers to the point-set
n_outliers = int(outliers_ratio * n_samples)
indices = np.arange(x_noise.shape[0])
np.random.shuffle(indices)
outlier_indices = indices[:n_outliers]

x_noise[outlier_indices] = 30 * np.random.random(size=(n_outliers, n_inputs))

# gaussian outliers
y_noise[outlier_indices] = 30 * np.random.normal(size=(n_outliers, n_outputs))

#--------------------------------

data = np.hstack((x_noise, y_noise))

ratio = 0.
model_m = 0.
model_c = 0.

# perform RANSAC iterations
for it in range(ransac_iterations):

  # pick up two random points
  n = 2

  all_indices = np.arange(x_noise.shape[0])
  np.random.shuffle(all_indices)

  indices_1 = all_indices[:n]
  indices_2 = all_indices[n:]

  maybe_points = data[indices_1, :]
  test_points = data[indices_2, :]

  # find a line model for these points
  m, c = find_line_model(maybe_points)

  x_list = []
  y_list = []
  num = 0

  # find orthogonal lines to the model for all testing points
  for ind in range(test_points.shape[0]):

    x0 = test_points[ind, 0]
    y0 = test_points[ind, 1]

    # find an intercept point of the model with a normal from point (x0,y0)
    x1, y1 = find_intercept_point(m, c, x0, y0)

    # distance from point to the model
    dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # check whether it's an inlier or not
    if dist < ransac_threshold:
      x_list.append(x0)
      y_list.append(y0)
      num += 1

  x_inliers = np.array(x_list)
  y_inliers = np.array(y_list)

  # in case a new model is better - cache it
  if num / float(n_samples) > ratio:
    ratio = num / float(n_samples)
    model_m = m
    model_c = c

  print
  '  inlier ratio = ', num / float(n_samples)
  print
  '  model_m = ', model_m
  print
  '  model_c = ', model_c

  # plot the current step
  ransac_plot(it, x_noise, y_noise, m, c, False, x_inliers, y_inliers, maybe_points)

  # we are done in case we have enough inliers
  if num > n_samples * ransac_ratio:
    print
    'The model is found !'
    break

# plot the final model
ransac_plot(0, x_noise, y_noise, model_m, model_c, True)

print
'\nFinal model:\n'
print
'  ratio = ', ratio
print
'  model_m = ', model_m
print
'  model_c = ', model_c