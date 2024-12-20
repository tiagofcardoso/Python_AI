import cv2
import numpy as np
import os

ovar = os.path.join(os.path.dirname(__file__), 'Ovar.jpg')

img =  cv2.imread(ovar, cv2.IMREAD_GRAYSCALE) #0

cv2.imwrite('ovar.png', img)
