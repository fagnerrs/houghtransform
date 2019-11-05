import sys
import matplotlib.pyplot as plt

def find_line_model(points):
  """ find a line model for the given points
  :param points selected points for model fitting
  :return line model
  """

  # [WARNING] vertical and horizontal lines should be treated differently
  #           here we just add some noise to avoid division by zero

  # find a line model for these points
  m = (points[1, 1] - points[0, 1]) / (
    points[1, 0] - points[0, 0] + sys.float_info.epsilon)  # slope (gradient) of the line
  c = points[1, 1] - m * points[1, 0]  # y-intercept of the line

  return m, c


def find_intercept_point(m, c, x0, y0):
  """ find an intercept point of the line model with
      a normal from point (x0,y0) to it
  :param m slope of the line model
  :param c y-intercept of the line model
  :param x0 point's x coordinate
  :param y0 point's y coordinate
  :return intercept point
  """

  # intersection point with the model
  x = (x0 + m * y0 - m * c) / (1 + m ** 2)
  y = (m * x0 + (m ** 2) * y0 - (m ** 2) * c) / (1 + m ** 2) + c

  return x, y

def ransac_plot(n, x, y, m, c, final=False, x_in=(), y_in=(), points=()):
  """ plot the current RANSAC step
  :param n      iteration
  :param points picked up points for modeling
  :param x      samples x
  :param y      samples y
  :param m      slope of the line model
  :param c      shift of the line model
  :param x_in   inliers x
  :param y_in   inliers y
  """

  fname = "images/figure_" + str(n) + ".png"
  line_width = 1.
  line_color = '#0080ff'
  title = 'iteration ' + str(n)

  if final:
    fname = "images/final.png"
    line_width = 3.
    line_color = '#ff0000'
    title = 'final solution'

  plt.figure("Ransac", figsize=(15., 15.))

  # grid for the plot
  grid = [min(x) - 10, max(x) + 10, min(y) - 20, max(y) + 20]
  plt.axis(grid)

  minx = int(min(x) - 10)
  maxx = int(max(x) + 10)

  for i in range(minx, maxx, 5):
    print('i', i)

  # put grid on the plot
  plt.grid(b=True, which='major', color='0.75', linestyle='--')
  plt.xticks([i for i in range(minx, maxx, 5)])

  minx = int(min(x) - 20)
  maxx = int(max(x) + 20)
  plt.yticks([i for i in range(minx, maxx, 10)])

  # plot input points
  plt.plot(x[:, 0], y[:, 0], marker='o', label='Input points', color='#00cc00', linestyle='None', alpha=0.4)

  # draw the current model
  plt.plot(x, m * x + c, 'r', label='Line model', color=line_color, linewidth=line_width)

  # draw inliers
  if not final:
    plt.plot(x_in, y_in, marker='o', label='Inliers', linestyle='None', color='#ff0000', alpha=0.6)

  # draw points picked up for the modeling
  if not final:
    plt.plot(points[:, 0], points[:, 1], marker='o', label='Picked points', color='#0000cc', linestyle='None',
             alpha=0.6)

  plt.title(title)
  plt.legend()
  plt.savefig(fname)
  plt.close()