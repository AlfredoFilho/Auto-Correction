# -*- coding: utf-8 -*-
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils
from keras.layers import Dense
from keras import backend as K
import numpy as np
import cv2
import os

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

def model():
    model = Sequential()

    #30 Features Maps com Kernel de 5x5 - Funcão de Ativação: relu
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(15, (3, 3), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    #Desligar neurônios para evitar overfitting(Modelo se ajustou demais aos dados)
    #20%
    model.add(Dropout(0.5))

    
    #Converte as camadas convolucionais em um longo vetor (transposta)
    model.add(Flatten())

    
    #Camadas de neurônios. 128 -> 64 -> 32 -> 10
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax', name='predict'))
    #Softmax para obter a distribuição de probabilidades

    
    #Consolidar arquitetura
    #Função de perda: categorical_crossentropy
    #Otimizador: adam
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

seed = 7
np.random.seed(seed)

x_train, y_train, x_test, y_test = load_data('datasetsMnist/classicMnist.npz')

#Numero de classes de predição (10)
num_classes = y_test.shape[1]

# O model será exportado para este arquivo
filename='mnistneuralnet.h5'

model = model()
model.summary()

# Verifica se já existe um modelo treinado e exportado para um arquivo .h5.
# Um novo modelo será treinado, caso este arquivo não exista.
if not os.path.exists('./{}'.format(filename) ):
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200)
    model.save_weights(filename)
else:
    # carrega um modelo previamente treinado
    model.load_weights('./{}'.format(filename) )

scores = model.evaluate(x_test, y_test, verbose=0)
print("\nAcurácia: %.2f%%" % (scores[1]*100))

img_pred = cv2.imread("imgsNumbers/number-five.png", 0)
plt.imshow(img_pred, cmap='gray')

#Deixar imagem no padrão do Mnist
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
