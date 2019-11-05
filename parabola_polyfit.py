import numpy as np
import matplotlib.pyplot as plt

points = np.array ([(0.05, 0.957), (0.12, 0.851), (0.15, 0.832), (0.30, 0.720),
(0.45, 0.583), (0.70, 0.378), (0.84, 0.295), (1.05, 0.156)])

points = np.array([(-4, 14), (-3,7), (-2,2), (-1, -0.9, ), (0, -1.9), (1, -0.9), (2, 2), (3,7), (4,14)])
x = points[:,0]
y = points[:,1]

a = np.polyfit(x, y, 2)
b = np.poly1d(a)
print(a)
print (b)

plt.plot(x,y)
plt.plot(x,b(x))
plt.show()