# -*- coding: utf-8 -*-
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.python.util import deprecation
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers import GlobalMaxPooling2D
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils
from keras.layers import Dense
from keras import backend as K
import numpy as np
import requests
import cv2
import os

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
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/filesDownload/MnistKeras.npz',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/filesDownload/myNumbers/number-one.png',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/filesDownload/myNumbers/number-two.png',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/filesDownload/myNumbers/number-three.png',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/filesDownload/myNumbers/number-four.png',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/filesDownload/myNumbers/number-five.png',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/filesDownload/myNumbers/number-six.png',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/filesDownload/myNumbers/number-seven.png',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/filesDownload/myNumbers/number-eight.png',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/filesDownload/myNumbers/number-nine.png'
            ]
    
    print('Download files...')
    
    if((os.path.isdir("files")) == False):
        os.mkdir('files')
    
    os.chdir('files')
        
    for url in urlsDownload:
        nameFile = url.split('/')[-1]
        if (os.path.exists(nameFile) == False):
            download_file(url)
    
    os.chdir("..")
    
    print('-Finish download-')

def load_data(path):

    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

        x_train = x_train / 255
        x_test = x_test / 255

        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test

downloadFiles()
np.random.seed(2019)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
x_train, y_train, x_test, y_test = load_data('files/MnistKeras.npz')

class Model:
    def __init__(self):
        self.model = self.getModel()

    def defineTrain(self, name, epochs, batch_size, nameCallback, patienceCallback):
        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size
        self.nameCallback = nameCallback
        self.patienceCallback = patienceCallback
        
    def get_callbacks(self, name, patience_lr):
        mcp_save = ModelCheckpoint(name + '.h5', save_best_only = True,
                                         monitor = 'val_accuracy', mode='max')
        reduce_lr_loss = ReduceLROnPlateau(monitor = 'loss', factor = 0.1,
                                           patience = patience_lr,
                                           verbose = 1, min_delta = 1e-4,
                                           mode = 'min')
        early_stop = EarlyStopping(monitor = 'val_accuracy', min_delta = 0, patience = 10,
                                   verbose = 0, mode = 'max', baseline = None)
        csv_logger = CSVLogger(name + '_log.csv', append = True, separator = ';')
        return [reduce_lr_loss, early_stop, csv_logger, mcp_save]
    
    def loadModel(self, nameLoad):
        self.model.load_weights(nameLoad)
        
    def trainModel(self):
        self.model.summary()
                
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  epochs= self.epochs, batch_size = self.batch_size,
                  callbacks = self.get_callbacks(self.nameCallback, self.patienceCallback))
        
        self.model.save_weights(self.name)
        
    def printAcc(self):
        scores = self.model.evaluate(x_test, y_test, verbose = 0)
        print("\nAcc: %.2f%%" % (scores[1]*100))
    
    def testModel(self):
        images = ['files/number-one.png', 'files/number-two.png', 'files/number-three.png', 'files/number-four.png', 'files/number-five.png',
              'files/number-six.png', 'files/number-seven.png', 'files/number-eight.png', 'files/number-nine.png'
              ]

        for image in images:
            img_pred = cv2.imread(image, 0)
            plt.imshow(img_pred, cmap='gray')
    
            if img_pred.shape != [28,28]:
                img2 = cv2.resize(img_pred, (28, 28))
                img_pred = img2.reshape(28, 28, -1)
            else:
                img_pred = img_pred.reshape(28, 28, -1)
    
            img_pred = ~img_pred
            img_pred = img_pred.reshape(1, 28, 28, 1).astype('float32')

            img_pred = img_pred/255.0
    
            pred = self.model.predict_classes(img_pred)
            pred_proba = self.model.predict_proba(img_pred)
            pred_proba = "%.2f%%" % (pred_proba[0][pred]*100)
            print(pred[0], " com confiança de ", pred_proba)
            plt.show()
        
    def getModel(self):

        num_classes = 10
        model = Sequential()

        model.add(Conv2D(128, (5, 5), input_shape = (28, 28, 1), activation = 'relu'))
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

brain = Model()
brain.loadModel('files/Brain_SGD_1024.h5')
#brain.defineTrain(name = 'Brain.h5', epochs = 1000, batch_size = 32, nameCallback = 'BrainCallback', patienceCallback = 10)
#brain.trainModel()
brain.testModel()
brain.printAcc()
