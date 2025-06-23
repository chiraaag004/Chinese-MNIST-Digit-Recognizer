import numpy as np
from layer import Layer
from activation import Activation

class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)
        
        def relu_prime(x):
            return (x > 0).astype(float)
        
        super().__init__(relu, relu_prime)

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class Softmax(Layer):
    def forward(self, input):
        expZ = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = expZ / np.sum(expZ, axis=1, keepdims=True)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient
