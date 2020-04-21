# -*- coding: utf-8 -*-
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers import GlobalMaxPooling2D
from keras.models import model_from_json
from matplotlib import pyplot as plt
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dropout
from keras.utils import np_utils
from keras.layers import Dense
import keras_metrics
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
            'https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/OldFiles/Model/datasets/MnistKeras.npz'
            ]
    
    print('\nDownload files...')
    
    if((os.path.isdir("Files")) == False):
        os.mkdir('Files')
    
    os.chdir('Files')
        
    for url in urlsDownload:
        nameFile = url.split('/')[-1]
        if (os.path.exists(nameFile) == False):
            download_file(url)
    
    os.chdir("..")
    
    print('...Finish download\n')

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
x_train, y_train, x_test, y_test = load_data('Files/MnistKeras.npz')


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

    
    def loadModel(self, h5NameLoad):

        self.model.load_weights(h5NameLoad)

        
    def trainModel(self):

        self.model.summary()
        self.model.fit(x_train, y_train, validation_data = (x_test, y_test),
                  epochs = self.epochs, batch_size = self.batch_size,
                  callbacks = self.get_callbacks(self.nameCallback, self.patienceCallback))
        
        self.model.save_weights(self.name)

        
    def printAcc(self):

        scores = self.model.evaluate(x_test, y_test, verbose = 0)
        print("\nAcc in MnistKeras: %.2f%%" % (scores[1]*100))

    def showImage(self, image):
        
        import matplotlib.pyplot as plt

        plt.figure(num='Press Q for quit')
        plt.rcParams["keymap.quit"] = "q"
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

    
    def testModel(self):

        images = ['Files/number-one.png', 'Files/number-two.png', 'Files/number-three.png', 'Files/number-four.png', 'Files/number-five.png',
              'Files/number-six.png', 'Files/number-seven.png', 'Files/number-eight.png', 'Files/number-nine.png']

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
            print(pred[0], " - ", pred_proba)
            plt.show()

        
    def getModel(self):

        num_classes = 10
        optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0)

        model = Sequential()

        model.add(Conv2D(128, (3, 3), input_shape = (28, 28, 1), activation = 'relu', padding = 'same'))
        model.add(AveragePooling2D(pool_size = (2, 2), strides = 2))
        model.add(Dropout(0.2))

        model.add(BatchNormalization())

        model.add(Conv2D(256, (3, 3), input_shape = (28, 28, 1), activation = 'relu', padding = 'same'))
        model.add(AveragePooling2D(pool_size = (2, 2), strides = 2))
        model.add(Dropout(0.2))

        model.add(BatchNormalization())
        
        model.add(Conv2D(512, (3, 3), input_shape = (28, 28, 1), activation = 'relu', padding = 'same'))
        model.add(AveragePooling2D(pool_size = (2, 2), strides = 2))
        model.add(Dropout(0.3))

        model.add(BatchNormalization())            
        model.add(GlobalMaxPooling2D())
            
        model.add(Dense(1024, activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation = 'softmax', name = 'predict'))
        
        model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy', keras_metrics.precision(), keras_metrics.recall()])
        
        from keras.models import model_from_json
        model_json = model.to_json()
        with open("modelBrabo.json", "w") as json_file:
            json_file.write(model_json)

        return model

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus

brain = Model()
brain.loadModel('Expert.h5')
brain.defineTrain(name = 'BrainLastEpoch.h5', epochs = 1000, batch_size = 128, nameCallback = 'Expert', patienceCallback = 10)
brain.trainModel()
#brain.testModel()
brain.printAcc()

#example = https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py