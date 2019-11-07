import cv2
import numpy as np

modelo = cv2.imread('ModeloPNG.png')

def findContours():
    
    img_gray = cv2.cvtColor(modelo, cv2.COLOR_RGB2GRAY)
    contours, hierarchy = cv2.findContours(~img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def cutImage (polygonExternalPoints):
    # Define the corners of the target rectangle
    width, height = 680, 966
    cutOutPoints = np.zeros(polygonExternalPoints.shape, dtype=np.float32)
    cutOutPoints[0] = (0, 0)
    cutOutPoints[1] = (0, height)
    cutOutPoints[2] = (width, height)
    cutOutPoints[3] = (width, 0)

    transformationMatrix = cv2.getPerspectiveTransform(polygonExternalPoints.astype(np.float32), cutOutPoints)

    # Apply perspective transformation
    bigRect = cv2.warpPerspective(modelo, transformationMatrix, (width, height))
    return bigRect
    

def extractTemplateRectangle():
    # Find contour of the biggest rectangle
    contours = findContours()
    templateRectangle = max(contours, key = cv2.contourArea)

    ## Stract the polygon points
    epsilonCurve = 0.01 * cv2.arcLength(templateRectangle, True)
    polygonExternalPoints = cv2.approxPolyDP(templateRectangle, epsilonCurve, True)

    bigRect = cutImage(polygonExternalPoints)
    return bigRect
    
bigRect = extractTemplateRectangle()
cv2.imshow('Modelo', bigRect)
cv2.waitKey(0)