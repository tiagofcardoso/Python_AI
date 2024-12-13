import cv2
import os

ovar = os.path.join(os.path.dirname(__file__), 'Ovar.jpg')
img = cv2.imread(ovar, cv2.IMREAD_COLOR) #1

print(f'Shape: {img.shape}')
print(f'Size: {img.size} pixeis')
print(f'Tipo da imagem: {img.dtype}')

ROI = img[300:400, 400:600]
img[0:100, 0:200] =ROI

cv2.imshow("img", img)

cv2.waitKey()
cv2.destroyAllWindows()