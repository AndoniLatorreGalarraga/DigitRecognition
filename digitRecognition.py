import numpy as np
import random
from Modules import neuralNetwork as nn
from Modules import mnist

def main():
    trainImages, trainLabels, testImages, testLabels= mnist.getData()
    net = nn.NeuralNetwork([784,30,10])
    trainData = [(i, o) for i, o in zip(trainImages, trainLabels)]
        
    epochs = 30
    for n in range(epochs):
        random.shuffle(trainData)
        for i in range(0, len(trainData), 10):
            net.batch(trainData[i:i+10], lRate = 0.5)
        print('Progress: ', n + 1, '/', epochs, sep = '')

        correct, total = 0, 0
        for (image, label) in zip(testImages, testLabels):
            if result(net.compute(image)) == label:
                correct += 1
            total += 1
        print('Results: ', correct, '/', total, sep = '')
    
    net.save()

def result(array):
    list = array.T.tolist()[0]
    return list.index(max(list))

if __name__ == '__main__':
    main()