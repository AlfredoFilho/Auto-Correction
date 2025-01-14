import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
import cv2
import json
import argparse
import numpy as np
from pdf2image import convert_from_path
from media.modules import image as processimage
from media.modules import number as processnumber
from media.modules import model as processmodel

#hide some system outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
listSixPredictions = []

class AutoCorrection:


    def __init__(self, pathToLoadData):
        self.pathToLoadData = pathToLoadData
        self.model = processmodel.getModel()
        self.loadData()

        
    def readJson(self):
		
        with open('media/coordinates.json') as fileJson:
            coordinatesJson = json.load(fileJson)
		
            return coordinatesJson


    def loadData(self):

        #read json for get coordinates
        coordinatesJson = self.readJson()
		
        if self.pathToLoadData[-4:] == '.png':
			
            #read image from path
            imageLoad = cv2.imread(self.pathToLoadData)
            self.resolveTemplate(imageLoad, coordinatesJson)

        elif self.pathToLoadData[-4:] == '.pdf':
            
            imagesPDF = convert_from_path(self.pathToLoadData)
            
            for image in imagesPDF:
                image = np.asarray(image)
                self.resolveTemplate(image, coordinatesJson)
                
        else:
            return print('\nO caminho/arquivo não é aceito. Somente extensão PDF ou PNG.\nExemplo: caminho/para/arquivo.pdf\n')


    def resolveTemplate(self, image, coordinatesJson):

        #get biggest reactangle
        bigRect = processimage.getBigRect(image)
        
        #resolve
        # self.resolveAlternatives(coordinatesJson, bigRect)
        self.resolveOthers(coordinatesJson, bigRect)
        self.resolveRA(coordinatesJson, bigRect)


    def resolveAnternative(self, question, listFiveRect):

        answers = { 0 : "A", 1 : "B", 2 : "C", 3 : "D", 4 : "E" }
        listAverages = []

        for rect in listFiveRect:
            #obtain average of each alternative of the question
			
            average = np.average(rect)
            listAverages.append(average)

        #get position for minimal average (marked question has the lowest average)
        answerMarked = listAverages.index(min(listAverages))

        print(f'Alternativa {question}:', answers[answerMarked])
        if int(question) % 10 == 0:
            print("")


    def resolveAlternatives(self, coordinatesJson, image):

        #dimensions of Alternatives square
        height = 9
        width = 30

        listFiveRect = []

        #Get Alternatives keys from JSON
        alternatives = list(coordinatesJson['Alternatives'].keys())

        for question in alternatives:
            for square in coordinatesJson['Alternatives'][question]:
                x = coordinatesJson['Alternatives'][question][square][0]
                y = coordinatesJson['Alternatives'][question][square][1]

                croppedRect = processimage.cropImage(x, y, width, height, image)
				
                listFiveRect.append(croppedRect)

                if len(listFiveRect) == 5:
                    self.resolveAnternative(question, listFiveRect)
                    listFiveRect = []


    def predictNumber(self, question, imageNumber):

        global listSixPredictions

        imageGray = cv2.cvtColor(imageNumber, cv2.COLOR_RGB2GRAY)
        imageProcessed = processnumber.processNumber(imageGray)
        imageReshape = processnumber.reshape(imageProcessed)

        prediction, percentage = processmodel.predictNumber(self.model, imageReshape)

        # if(question == "31"):
        # print(prediction[0], " -> ", percentage)
        # processimage.showImage(imageNumber)
        processimage.showImage(imageProcessed)

        listSixPredictions.append(prediction[0])
        if len(listSixPredictions) == 6:
            print('Alternativa ' + question +': ' + ''.join(str(s) for s in listSixPredictions))
            print()
            listSixPredictions = []
        


    def resolveRA(self, coordinatesJson, image):

        #dimensions of RA square
        height = 29
        width = 25

        for square in coordinatesJson['RA']:
            x = coordinatesJson['RA'][square][0]
            y = coordinatesJson['RA'][square][1]

            #crop image
            croppedSquare = processimage.cropImage(x, y, width, height, image)
            self.predictNumber(question = 'RA', imageNumber = croppedSquare)


    def resolveOthers(self, coordinatesJson, image):

        #dimensions of OthersAnswers square
        height = 29
        width = 25

        #Get OthersAnswers keys from JSON
        others = list(coordinatesJson['Others'].keys())

        for question in others:
            for square in coordinatesJson['Others'][question]:
                x = coordinatesJson['Others'][question][square][0]
                y = coordinatesJson['Others'][question][square][1]

                #crop image
                croppedSquare = processimage.cropImage(x, y, width, height, image)
                self.predictNumber(question = question, imageNumber = croppedSquare)


#Define argument essential
parser = argparse.ArgumentParser(description='Auto Corretion Tests - AlfredoFilho')
parser.add_argument('-f','--file', help = 'Path to file. Example: path/to/file.pdf', required=True)
args = vars(parser.parse_args())

AutoCT = AutoCorrection(pathToLoadData = args['file'])