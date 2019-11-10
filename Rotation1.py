from HoughParabola import *

theta = 360
R1 = np.array([[round(np.cos(theta)), round(np.sin(theta))],
              [round(-np.sin(theta)), round(np.cos(theta))]])

R = np.array([[round(np.cos(theta)), round(np.sin(theta)), 0],
              [round(-np.sin(theta)), round(np.cos(theta)), 1],
              [0, 0, 1]])

x = np.array([2, 2, 1]).T
x1 = np.array([2, 2])

print(R)
print(x)
print(R.dot(x))
print(R1.dot(x1))


x = 2
y = 2
ox = 0
oy = 0
qx = ox + np.cos(theta) * (x - ox) + np.sin(theta) * (y - oy)
qy = oy + -np.sin(theta) * (x - ox) + np.cos(theta) * (y - oy)

print('{0} - {1}', qx, qy)

