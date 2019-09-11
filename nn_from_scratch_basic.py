# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import numpy as np

# def relu(x):
#     return max(0,x)

def sigmoid(x):
    return 1 / (1+np.e**-x)

def sigmoid_deriviative(x):
    return x * (1 - x)


x = np.array([[1,2,3],
              [3,1,8]])

y = np.array([[10],
             [24]])


class NueralNetwork:
    def __init__(self,x,y):
        self.input = x
        self.weights1 = np.random.randn(x.shape[1],10)
        self.weights2 = np.random.randn(10,1)
        self.y = y

    def fforward(self):
        self.layer1 = sigmoid(self.input @ self.weights1)
        self.layer2 = sigmoid(self.layer1 @ self.weights2)
        self.yhat = self.layer2


    def backprop(self):
        self.grad2 = self.weigths2 @ (2 * (self.yhat-self.y) * sigmoid_deriviative(self.yhat))
        
        self.grad1 = self.weights1 @  (2 * (self.yhat-self.y) * sigmoid_deriviative(self.yhat))
        
        
        self.weights1 = self.weights1 - self.grad


nn = NueralNetwork(x,y)

nn.fforward()

