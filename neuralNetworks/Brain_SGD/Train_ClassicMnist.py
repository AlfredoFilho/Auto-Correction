# -*- coding: utf-8 -*-
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import GlobalMaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils
from keras.layers import Dense
from keras import backend as K
import os
import numpy as np
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    #Definindo formato - (60000, 1, 28, 28)
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
    
    
    #Definindo formato - (10000, 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
    
    
    #Deixar no intervalo de 0 e 1, ao invés de 0 a 255
    x_train = x_train / 255
    x_test = x_test / 255

    
    #Transformar rótulos em valores categóricos
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    
    return x_train, y_train, x_test, y_test

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


def model():
    model = Sequential()

    model.add(Conv2D(256, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(1024, (3, 3), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(GlobalMaxPooling2D())

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name='predict'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return model

seed = 7
np.random.seed(seed)

x_train, y_train, x_test, y_test = load_data('classicMnist.npz')

#Numero de classes de predição (10)
num_classes = y_test.shape[1]

# O model será exportado para este arquivo
filename='mnistneuralnet.h5'

model = model()
model.summary()

# Verifica se já existe um modelo treinado e exportado para um arquivo .h5.
# Um novo modelo será treinado, caso este arquivo não exista.
if not os.path.exists('./{}'.format(filename) ):
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1000, batch_size=32, callbacks=get_callbacks("Brain", 10))
    model.save_weights(filename)
else:
    # carrega um modelo previamente treinado
    model.load_weights('./{}'.format(filename) )

scores = model.evaluate(x_test, y_test, verbose=0)
print("\nAcurácia: %.2f%%" % (scores[1]*100))
