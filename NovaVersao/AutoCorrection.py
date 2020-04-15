import cv2
import json
import numpy as np
from pdf2image import convert_from_path

import sys
sys.path.insert(1, 'Modules')
import ProcessImage
import ProcessNumber

class AutoCorrection:

    def __init__(self, pathToLoadData):
        self.pathToLoadData = pathToLoadData
        self.loadData()

        
    def loadData(self):

        #read json for get coordinates
        coordinatesJson = self.readJson()
		
        if self.pathToLoadData[-4:] == '.png':
			
            #read image from path
            imageLoad = cv2.imread(self.pathToLoadData)
            self.resolveTemplate(imageLoad, coordinatesJson)

        elif self.pathToLoadData[-4:] == '.pdf':
            
            imagesPDF = convert_from_path(self.pathToLoadData, 100)
            
            for image in imagesPDF:
                self.resolveTemplate(imageLoad, coordinatesJson)
                
        else:
            return print('\nO caminho/arquivo não é aceito. Somente extensão PDF ou PNG.\nExemplo: caminho/para/arquivo.pdf\n')


    def resolveTemplate(self, image, coordinatesJson):

        #get biggest reactangle
        bigRect = ProcessImage.getBigRect(image)
        
        #resolve
        self.resolveRA(coordinatesJson, bigRect)
        self.resolveAlternatives(coordinatesJson, bigRect)
        self.resolveOthers(coordinatesJson, bigRect)


    def readJson(self):
		
        coordinatesJson = {}

        with open('coordinates.json') as fileJson:
            coordinatesJson = json.load(fileJson)
		
        return coordinatesJson


    def resolveRA(self, coordinatesJson, image):

        #dimensions of RA square
        height = 31
        width = 29

        for square in coordinatesJson['RA']:
            x = coordinatesJson['RA'][square][0]
            y = coordinatesJson['RA'][square][1]

            #crop image
            croppedSquare = ProcessImage.cropImage(x, y, width, height, image)


    def resolveAnternative(self, question, listFiveRect):

        answers = {
				0 : "A",
				1 : "B",
				2 : "C",
				3 : "D",
				4 : "E"
        }

        listMedia = []

        for rect in listFiveRect:
            #obtain average of each alternative of the question
			
            average = np.average(rect)
            listMedia.append(average)

        #get position for minimal average (marked question has the lowest average)
        answerMarked = listMedia.index(min(listMedia))

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

                croppedRect = ProcessImage.cropImage(x, y, width, height, image)
				
                listFiveRect.append(croppedRect)

                if len(listFiveRect) == 5:
                    self.resolveAnternative(question, listFiveRect)
                    listFiveRect = []


    def resolveOthers(self, coordinatesJson, image):

        #dimensions of OthersAnswers square
        height = 31
        width = 29

        #Get OthersAnswers keys from JSON
        others = list(coordinatesJson['Others'].keys())

        for question in others:
            for square in coordinatesJson['Others'][question]:
                x = coordinatesJson['Others'][question][square][0]
                y = coordinatesJson['Others'][question][square][1]

                #crop image
                croppedSquare = ProcessImage.cropImage(x, y, width, height, image)

import argparse

parser = argparse.ArgumentParser(description='Auto Corretion Tests - AlfredoFilho')
parser.add_argument('-f','--file', help = 'Path to file. Example: path/to/file.pdf', required=True)
args = vars(parser.parse_args())

AutoCT = AutoCorrection(pathToLoadData = args['file'])