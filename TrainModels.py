# -*- coding: utf-8 -*-
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
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
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()

def downloadFiles():
    
    urlsDownload = [
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/neuralNetworks/1024/Brain_SGD_1024.h5',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/trainNeuralNet/datasetsMnist/classicMnist.npz',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/neuralNetworks/imgsNumbers/number-one.png',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/neuralNetworks/imgsNumbers/number-two.png',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/neuralNetworks/imgsNumbers/number-three.png',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/neuralNetworks/imgsNumbers/number-four.png',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/neuralNetworks/imgsNumbers/number-five.png',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/neuralNetworks/imgsNumbers/number-six.png',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/neuralNetworks/imgsNumbers/number-seven.png',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/neuralNetworks/imgsNumbers/number-eight.png',
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/neuralNetworks/imgsNumbers/number-nine.png'
            ]
    
    for url in urlsDownload:
        nameFile = url.split('/')[-1]
        if (os.path.exists(nameFile) == False):
            download_file(url)

def tests(model):
    images = ['number-one.png', 'number-two.png', 'number-three.png', 'number-four.png', 'number-five.png',
              'number-six.png', 'number-seven.png', 'number-eight.png', 'number-nine.png'
              ]

    for image in images:
      img_pred = cv2.imread(image, 0)
      plt.imshow(img_pred, cmap='gray')
      plt.show()
    
      if img_pred.shape != [28,28]:
          img2 = cv2.resize(img_pred, (28, 28))
          img_pred = img2.reshape(28, 28, -1)
      else:
          img_pred = img_pred.reshape(28, 28, -1)
    
      img_pred = ~img_pred
      img_pred = img_pred.reshape(1, 1, 28, 28).astype('float32')
    
      img_pred = img_pred/255.0
    
    
      pred = model.predict_classes(img_pred)
      pred_proba = model.predict_proba(img_pred)
      pred_proba = "%.2f%%" % (pred_proba[0][pred]*100)
      print(pred[0], " com confiança de ", pred_proba)


class loadModel:
    def __init__(self, name):
        self.name = name
        self.model = Model()
        self.model.load_weights(name)
    
    def testModel(self):
        tests(self.model)

class createModel:
    def __init__(self, name, epochs, batch_size, nameCallback, patienceCallback):
        self.name = name
        self.epochs = epochs
        self.batch_size = batch_size
        self.nameCallback = nameCallback
        self.patienceCallback = patienceCallback
    
    def get_callbacks(name, patience_lr):
        mcp_save = ModelCheckpoint(name+".h5", save_best_only=True,
                                         monitor='val_accuracy', mode='max')
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1,
                                           patience=patience_lr,
                                           verbose=1, epsilon=1e-4,
                                           mode='min')
        early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10,
                                   verbose=0, mode='max', baseline=None)
        csv_logger = CSVLogger(name + '_log.csv', append=True, separator=';')
        return [reduce_lr_loss, early_stop, csv_logger, mcp_save]
    
    def load_data(path):
        with np.load(path) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
    
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
        
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
        
        x_train = x_train / 255
        x_test = x_test / 255
    
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        
        return x_train, y_train, x_test, y_test
    
    def trainModel(self):
        model = Model()
        model.summary()
        
        x_train, y_train, x_test, y_test = self.load_data('classicMnist.npz')
        
        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  epochs= self.epochs, batch_size = self.batch_size,
                  callbacks = self.get_callbacks(self.nameCallback, self.patienceCallback))
        
        model.save_weights(self.name)
        
        scores = model.evaluate(x_test, y_test, verbose=0)
        print("\nAcc: %.2f%%" % (scores[1]*100))
        
def Model():
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #INFO, WARNING, and ERROR messages are not printed
np.random.seed(2019)

downloadFiles()

#create and train model
#brain = createModel(name = 'Brain.h5', epochs = 1000, batch_size = 32, nameCallback = 'Albelis', patienceCallback = 10)
#brain.trainModel()

#load model and tests
brainLoad = loadModel(name = 'Brain_SGD_1024.h5')
brainLoad.testModel()