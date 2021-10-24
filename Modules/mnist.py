import numpy as np
import idx2numpy

def getData():
    trainLabels = idx2numpy.convert_from_file('Data/train-labels.idx1-ubyte')
    trainImages = idx2numpy.convert_from_file('Data/train-images.idx3-ubyte')
    testLabels = idx2numpy.convert_from_file('Data/t10k-labels.idx1-ubyte')
    testImages = idx2numpy.convert_from_file('Data/t10k-images.idx3-ubyte')
    trainImagesReshape = []
    testImagesReshape = []
    for image in trainImages:
        trainImagesReshape.append(np.reshape(image,(784,1)))
    for image in testImages:
        testImagesReshape.append(np.reshape(image,(784,1)))
    trainLabelsArray = []
    for label in trainLabels:
        array = [0]*10
        array[label] = 1
        trainLabelsArray.append(np.array([array]).T)
    return trainImagesReshape, trainLabelsArray, testImagesReshape, testLabels

def printImage(array):
    array = np.reshape(array,(28, 28))
    light =  ''' .\'`^",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$'''
    for i in range(len(array)):
        s = ''
        for j in range(len(array[i])):
            print(light[round((69/255) * array[i][j])], end = '')
        print('')

def image(array):
    array = np.reshape(array,(28, 28))
    light =  ''' .\'`^",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$'''
    s = ''
    for i in range(len(array)):
        for j in range(len(array[i])):
            s += light[round((69/255) * array[i][j])]
        s += '\n'
    return s[:-2]