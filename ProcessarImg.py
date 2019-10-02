import cv2

imageOriginal = cv2.imread('ModeloGabarito\VetorialGabarito.png')
resized_image = cv2.resize(imageOriginal, (800, 800))

xCoordDict = {
'01-A_E' : [83, 121, 159, 196, 234],
'11-A_E' : [329, 367, 405, 442, 480],
'21-A_E' : [575, 613, 651, 688, 725]
}

#for key in xCoordDict:
#    
#    yCoordRect = 207
#    xListCoord = xCoordDict[key]
#    
#    for i in range(10):
#        for xCoordRect in xListCoord:
#            
#            crop = resized_image[yCoordRect:yCoordRect + 9, xCoordRect:xCoordRect + 30]
#            cv2.imshow('Image', crop)
#            cv2.waitKey(0)
#            
#        yCoordRect = yCoordRect + 20 #Next line

RA = [294, 333, 371, 409, 447, 485]
yCoordRA = 62
height = 31
width = 29

#for xCoordRA in RA:
#    
#    crop = resized_image[yCoordRA: yCoordRA + height, xCoordRA:xCoordRA + width]
#    cv2.imshow('Image', crop)
#    cv2.waitKey(0)