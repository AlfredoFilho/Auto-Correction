import cv2
import numpy as np

img = cv2.imread('ModeloGabarito\VetorialGabarito.png')
edged = cv2.Canny(img, 30, 200) 

contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(img, contours, -1, (0, 255, 0), 1) 
cv2.imshow("Keypoints", img)
cv2.waitKey(0)
cv2.destroyAllWindows()