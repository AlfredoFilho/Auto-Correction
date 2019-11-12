#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import os
import requests
import numpy as np
from copy import deepcopy

def cropImage(xCoord, yCoord, height, width):
    
    global resized_image
    crop = resized_image[yCoord:yCoord + height, xCoord:xCoord + width]
    cv2.imshow('Image', crop)
    cv2.waitKey(0)


def mask(xCoord, yCoord, height, width):
    
    global maskImage
    maskImage[yCoord:yCoord + height, xCoord:xCoord + width] = resized_image[yCoord:yCoord + height, xCoord:xCoord + width]


def multipleChoice():

    global maskImage, maskAux
    xVertexDict = {'01_Vertex_ABCDE': [83, 121, 159, 196, 234],
                   '11_Vertex_ABCDE': [329, 367, 405, 442, 480],
                   '21_Vertex_ABCDE': [575, 613, 651, 688, 725]}

    widthRect = 30
    heightRect = 9

    for key in xVertexDict:
        yCoordRect = 207
        xListCoord = xVertexDict[key]

        for i in range(10):  # 10 questions
            maskImage = deepcopy(maskAux)
            for xCoordRect in xListCoord:
                mask(xCoordRect, yCoordRect, heightRect, widthRect)

                cv2.imshow('Image', maskImage)
                cv2.waitKey(0)
            yCoordRect = yCoordRect + 20  # Next Question
    cv2.destroyAllWindows()


def RA():

    global maskImage, maskAux    
    yCoordRA = 62
    widthSquare = 29
    heightSquare = 31
    xVertexRA = [294, 333, 371, 409, 447, 485]

    maskImage = deepcopy(maskAux)

    for xCoordRA in xVertexRA:
        mask(xCoordRA, yCoordRA, heightSquare, widthSquare)

        cv2.imshow('Image', maskImage)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def otherAnswers():

    global maskImage, maskAux
    xVertexDict = {'31_Vertex_ABCDE': [129, 167, 205, 243, 280, 318],
                   '36_Vertex_ABCDE': [469, 507, 544, 582, 620, 658]}

    widthSquare = 29
    heightSquare = 31

    for key in xVertexDict:
        yCoord = 443
        xListCoord = xVertexDict[key]

        for i in range(5):  # 5 Questions
            maskImage = deepcopy(maskAux)
            for xCoord in xListCoord:
                mask(xCoord, yCoord, heightSquare, widthSquare)

                cv2.imshow('Image', maskImage)
                cv2.waitKey(0)
            yCoord = yCoord + 44  # Next Question
            
    cv2.destroyAllWindows()

imageOriginal = cv2.imread('TemplateTeste_v1.png')
resized_image = cv2.resize(imageOriginal, (800, 800))
maskImage = np.zeros(resized_image.shape, np.uint8)
maskAux = deepcopy(maskImage)

multipleChoice()
RA()
otherAnswers()