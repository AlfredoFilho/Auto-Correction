import cv2
import matplotlib.pyplot as plt


def findContours(image):
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    contours, hierarchy = cv2.findContours(~img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def getBigRect(imageLoad):

    contours = findContours(imageLoad)
    templateRectangle = max(contours, key = cv2.contourArea)

    ## Stract the polygon points
    epsilonCurve = 0.01 * cv2.arcLength(templateRectangle, True)
    polygonExternalPoints = cv2.approxPolyDP(templateRectangle, epsilonCurve, True)

    x,y,w,h = cv2.boundingRect(polygonExternalPoints)
    bigRect = imageLoad[y:y+h, x:x+w]
    bigRect = cv2.resize(bigRect, (800, 1000))

    return bigRect


def cropImage(x, y, width, height, image):

    croppedImage = image[y:y + height, x:x + width]

    return croppedImage


def drawCrop(x, y, width, height, image):

    cv2.rectangle(image, (x, y), (x + width, y + height), (0,255,0), 2)
    showImage(image)


def showImage(image):
    
    plt.figure(num='Press Q for quit')
    plt.rcParams["keymap.quit"] = "q"
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()