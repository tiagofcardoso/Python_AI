import cv2
import numpy as np
import os

ovar = os.path.join(os.path.dirname(__file__), 'Ovar.jpg')

img = cv2.imread(ovar, cv2.IMREAD_COLOR)

height, width = img.shape[:2]

#Resize
img_scale_up = cv2.resize(img,None, fx=2.0, fy=2, interpolation= cv2.INTER_CUBIC)
img_scale_down = cv2.resize(img,None, fx=0.5, fy=1, interpolation= cv2.INTER_CUBIC)

img_pyr_down =  cv2.pyrDown(img)
img_pyr_up =  cv2.pyrUp(img)

# Translação (Translation)
tx, ty =  100,50
matrix_translation = np.float32([[1, 0, tx],
                                 [0, 1, ty]])
img_translation = cv2.warpAffine(img, matrix_translation, (width,height))

# Rotação (Rotation)
center = (width//2, height //2)
angle = 45
scale=1.0
matrix_rotation = cv2.getRotationMatrix2D(center, angle, scale)
img_rotation = cv2.warpAffine(img, matrix_rotation, (width,height))

# Transformação Afim (Affine Transform)
pts1 = np.float32()
pts1 = np.float32([[50, 50],
                   [200, 50],
                   [50, 200]])
pts2 = np.float32([[10, 100],
                   [200, 50],
                   [100, 250]])
matrix_affine= cv2.getAffineTransform(pts1, pts2)
img_affine =  cv2.warpAffine(img, matrix_affine, (width,height))

# Transformação por Perspectiva (Perspective Transform)
pts1_persp = np.float32([[50,50],
                         [width-50,50],
                         [50,height-50],
                         [width-50,height-50]])
pts2_persp = np.float32([[0,0],
                         [300,0],
                         [0,300],
                         [300,300]])
M_perspective = cv2.getPerspectiveTransform(pts1_persp, pts2_persp)
img_perspective = cv2.warpPerspective(img, M_perspective, (300, 300))

# Exibir as imagens
cv2.imshow('Original', img)
cv2.imshow('Scale Up (2x)', img_scale_up)
cv2.imshow('Scale Down (0.5x)', img_scale_down)
cv2.imshow('PyrDown', img_pyr_down)
cv2.imshow('PyrUp', img_pyr_up)
cv2.imshow('Translation', img_translation)
cv2.imshow('Rotation 45 graus', img_rotation)
cv2.imshow('Affine Transformation', img_affine)
cv2.imshow('Perspective Transformation', img_perspective)

cv2.waitKey(0)
cv2.destroyAllWindows()