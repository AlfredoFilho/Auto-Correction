from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers import GlobalMaxPooling2D
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.layers import Dense
from copy import deepcopy
import numpy as np
import keras
import requests
import cv2
import os

from keras import backend as K
K.set_image_data_format('channels_first')

listResp = [None, 9999]
numberPred = ''

def download_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream = True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()

def downloadFiles():
    
    urlsDownload = [
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/models/1024/Brain_SGD_1024.h5',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/ImageTest.png'
            ]
    
    print('\nDownload files...')
    
    if((os.path.isdir("files")) == False):
        os.mkdir('files')
    
    os.chdir('files')
        
    for url in urlsDownload:
        nameFile = url.split('/')[-1]
        if (os.path.exists(nameFile) == False):
            download_file(url)
    
    os.chdir("..")
    
    print('...Finish download\n')

class GetAnswers:
    
    def __init__(self, bigRect):
        self.imageBigRect = bigRect
        self.resized_image = None
        self.maskImage = None
        self.maskAux = None
        self.model = self.Model()
        self.model.load_weights('files/Brain_SGD_1024.h5')
        self.preProcess()
        self.RA()
        self.otherAnswers()

    def preProcess(self):
        self.resized_image = cv2.resize(self.imageBigRect, (800, 800))
        self.maskImage = np.zeros(self.resized_image.shape, np.uint8)
        self.maskAux = deepcopy(self.maskImage)
        self.multipleChoice()
        
    
    def cropMChoice(self, xCoord, yCoord, height, width, cont):
        
        #crop = self.resized_image[yCoord:yCoord + height, xCoord:xCoord + width]
        media = np.average(self.resized_image[yCoord:yCoord + height, xCoord:xCoord + width])
    
        if media < listResp[1]:
            if cont == 1:
                listResp[0] = 'A'
            if cont == 2:
                listResp[0] = 'B'
            if cont == 3:
                listResp[0] = 'C'
            if cont == 4:
                listResp[0] = 'D'
            if cont == 5:
                listResp[0] = 'E'
            
            listResp[1] = media
        
        #cv2.imshow('Image', crop)
        #cv2.waitKey(0)
    
    def cropNumber(self, xCoord, yCoord, height, width):
        global numberPred
        
        crop = self.resized_image[yCoord:yCoord + height, xCoord:xCoord + width]
        
        img_pred = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

        if img_pred.shape != [28,28]:
            img2 = cv2.resize(img_pred, (28, 28))
            img_pred = img2.reshape(28, 28, -1)
        else:
            img_pred = img_pred.reshape(28, 28, -1)

        img_pred = ~img_pred
        img_pred = img_pred.reshape(1, 1, 28, 28).astype('float32')

        img_pred = img_pred/255.0

        pred = self.model.predict_classes(img_pred)
        pred_proba = self.model.predict_proba(img_pred)
        pred_proba = "%.2f%%" % (pred_proba[0][pred]*100)
        #print(pred[0], " com confianca de ", pred_proba)
        numberPred = numberPred + str(pred[0]) 
        
    
    def multipleChoice(self):
    
        global maskImage, maskAux, listResp
        xVertexDict = {'01_Vertex_ABCDE': [83, 121, 159, 196, 234],
                       '11_Vertex_ABCDE': [329, 367, 405, 442, 480],
                       '21_Vertex_ABCDE': [575, 613, 651, 688, 725]}
    
        widthRect = 30
        heightRect = 9
        cont = 0
        contAlt = 0
    
        for key in xVertexDict:
            yCoordRect = 207
            xListCoord = xVertexDict[key]
            print('')
    
            for i in range(10):  # 10 questions
                self.maskImage = deepcopy(self.maskAux)
                for xCoordRect in xListCoord:
                    cont = cont + 1
                    self.cropMChoice(xCoordRect, yCoordRect, heightRect, widthRect, cont)
    
                yCoordRect = yCoordRect + 20  # Next Question
                contAlt = contAlt + 1
                cont = 0
                print('Alternativa ' + str(contAlt) + ': ' + str(listResp[0]))
                listResp = [None, 9999]
    
    def Model(self):
        num_classes = 10
        model = Sequential()
            
        model.add(Conv2D(128, (5, 5), input_shape=(1, 28, 28), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(256, (3, 3), input_shape=(1, 28, 28), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(512, (3, 3), input_shape=(1, 28, 28), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.3))
    
        model.add(GlobalMaxPooling2D())
        
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax', name='predict'))
    
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        
        return model
        

    def RA(self):
    
        print('')
        global maskImage, maskAux, numberPred   
        yCoordRA = 62
        widthSquare = 29
        heightSquare = 31
        xVertexRA = [294, 333, 371, 409, 447, 485]
    
        maskImage = deepcopy(self.maskAux)
    
        for xCoordRA in xVertexRA:
            self.cropNumber(xCoordRA, yCoordRA, heightSquare, widthSquare)
    
        print('RA: ' + str(numberPred))
        numberPred = ''
        
    def otherAnswers(self):

        global maskImage, maskAux, numberPred
        xVertexDict = {'31_Vertex_ABCDE': [129, 167, 205, 243, 280, 318],
                       '36_Vertex_ABCDE': [ 469, 507, 544, 582, 620, 658]}
    
        widthSquare = 29
        heightSquare = 31
        print('')
        contOA = 30
    
        for key in xVertexDict:
            yCoord = 443
            xListCoord = xVertexDict[key]
    
            for i in range(5):  # 5 Questions
                maskImage = deepcopy(self.maskAux)
                for xCoord in xListCoord:
                    self.cropNumber(xCoord, yCoord, heightSquare, widthSquare)
    
                yCoord = yCoord + 44  # Next Question
                contOA = contOA + 1
                print('Alternativa ' + str(contOA) + ': ' + str(numberPred))
                numberPred = ''
                print('')
                
    

class AutoCorrection:
    
    def __init__(self, image):
        self.image = image
        self.bigRect = None
        self.extractTemplateRectangle()
        self.resolveTemplate()
        

    def findContours(self):
        
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        contours, hierarchy = cv2.findContours(~img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    
    def cutImage(self, polygonExternalPoints):
        # Define the corners of the target rectangle
        width, height = 800, 800
        cutOutPoints = np.zeros(polygonExternalPoints.shape, dtype=np.float32)
        cutOutPoints[0] = (0, 0)
        cutOutPoints[1] = (0, height)
        cutOutPoints[2] = (width, height)
        cutOutPoints[3] = (width, 0)
    
        transformationMatrix = cv2.getPerspectiveTransform(polygonExternalPoints.astype(np.float32), cutOutPoints)
    
        # Apply perspective transformation
        self.bigRect = cv2.warpPerspective(image, transformationMatrix, (width, height))
        
    
    def extractTemplateRectangle(self):
        # Find contour of the biggest rectangle
        contours = self.findContours()
        templateRectangle = max(contours, key = cv2.contourArea)
    
        ## Stract the polygon points
        epsilonCurve = 0.01 * cv2.arcLength(templateRectangle, True)
        polygonExternalPoints = cv2.approxPolyDP(templateRectangle, epsilonCurve, True)
    
        self.cutImage(polygonExternalPoints)

    
    def viewBiggestRect(self):
        cv2.imshow('Biggest Rectangle', self.bigRect)
        cv2.waitKey(0)
        
        
    def resolveTemplate(self):
        asd = GetAnswers(self.bigRect)

downloadFiles()
image = cv2.imread('files/ImageTest.png')
prova = AutoCorrection(image)
prova.viewBiggestRect()