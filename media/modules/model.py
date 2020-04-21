from keras.models import model_from_json

def getModel():

    jsonFile = open('media/brainModel.json', 'r')
    loadedJson = jsonFile.read()
    jsonFile.close()
    model = model_from_json(loadedJson)

    model = loadModel(model)

    return model


def loadModel(model):

    model.load_weights('media/brain.h5')

    return model


def predictNumber(model, imageNumber):

    prediction = model.predict_classes(imageNumber)
    percentage = model.predict_proba(imageNumber)
    percentage = "%.2f%%" % (percentage[0][prediction]*100)

    return prediction, percentage