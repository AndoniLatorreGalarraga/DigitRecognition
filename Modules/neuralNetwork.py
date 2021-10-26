import random, math
import numpy as np
# from numpy.lib.npyio import load, save

def sigma(z):
    try:
        return 1 / (1 + math.exp(-z))
    except OverflowError:
        if z > 0:
            return 1
        return 0

sig = np.vectorize(sigma)

def dsigma(z):
    return sig(z) * (1 - sig(z))

dsig = np.vectorize(dsigma)

class NeuralNetwork():
    '''
    NeuralNetwork(layerSizes)
    .batch(data, lRate = 0.001)
    .compute(i)
    '''
    def __init__(self, layerSizes = [3, 2, 3]):
        self.weights = [np.random.randn(n, m) for n, m in zip(layerSizes[1:], layerSizes[:-1])]
        self.biases = [np.random.randn(n, 1) for n in layerSizes[1:]]
        self.layerSizes = layerSizes
        self.depth = len(layerSizes)
        self.prevGradW = [np.zeros((n, m)) for n, m in zip(self.layerSizes[1:], self.layerSizes[:-1])]
        self.prevGradB = [np.zeros((n, 1)) for n in self.layerSizes[1:]]
    
    def batch(self, data, lRate = 1, mFactor = 0):
        gradW = [np.zeros((n, m)) for n, m in zip(self.layerSizes[1:], self.layerSizes[:-1])]
        gradB = [np.zeros((n, 1)) for n in self.layerSizes[1:]]
        dataLen = 0
        for d in data:
            i , o = d
            dataLen += 1
            #propagate forward
            a = i
            aList, zList = [a], [None]
            for weights, biases in zip(self.weights, self.biases):
                z = np.dot(weights, a) + biases
                zList.append(z)
                a = sig(z)
                aList.append(a)
            #propagate backward
            dW= [np.zeros((n, m)) for n, m in zip(self.layerSizes[1:], self.layerSizes[:-1])]
            dB = [np.zeros((n, 1)) for n in self.layerSizes[1:]]
            dA = [np.zeros((n, 1)) for n in self.layerSizes]
            dA[-1] = aList[-1] - o
            dB[-1] = np.multiply(dA[-1], dsig(zList[-1]))
            dW[-1] = np.dot(dB[-1], aList[-2].T)
            for l in range(self.depth - 2, 0, -1):
                dA[l] = np.zeros((self.layerSizes[l],1))
                for k in range(self.layerSizes[l+1]):
                    scalar = np.asscalar((dA[l+1][k] * dsigma(zList[l+1][k])))
                    dA[l] = dA[l] + np.array([self.weights[l][k]]).T * scalar
                dB[l-1] = np.multiply(dA[l], dsig(zList[l]))
                dW[l-1] = np.dot(dB[l-1], aList[l-1].T)
            for l in range(0, self.depth-1):
                gradB[l] = gradB[l] + dB[l]
                gradW[l] = gradW[l] + dW[l]
        #update weights and biases
        for l in range(0, self.depth-1):
            self.weights[l] = self.weights[l] - (gradW[l] * (lRate/dataLen) - mFactor * self.prevGradW[l])
            self.biases[l] = self.biases[l] - (gradB[l] * (lRate/dataLen) - mFactor * self.prevGradB[l])
        self.prevGradW, self.prevGradB = gradW, gradB
    
    def compute(self, i):
        for w, b in zip(self.weights, self.biases):
            i = sig(np.dot(w, i) + b)
        return i
    
    def save(self):
        layerSizes = np.asarray(np.array(self.layerSizes))
        np.save('Data/weights', self.weights)
        np.save('Data/biases', self.biases)
        np.save('Data/layerSizes', layerSizes)
    
    def load(self):
        self.weights = np.load('Data/weights.npy', allow_pickle = True)
        self.biases = np.load('Data/biases.npy', allow_pickle = True)
        layerSizes = np.load('Data/layerSizes.npy', allow_pickle = True)
        self.layerSizes = [i for i in layerSizes]
        self.depth = len(self.layerSizes)
