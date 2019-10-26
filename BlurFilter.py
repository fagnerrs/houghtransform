import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/exemplo1.jpg')

#blur = cv2.blur(img,(35,35))
#blur = cv2.GaussianBlur(img,(35,35),0)
#blur = cv2.medianBlur(img,35)
blur = cv2.bilateralFilter(img,35,150,200)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()