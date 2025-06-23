import numpy as np
from layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        batch_size = input.shape[0]
        return input.reshape(batch_size, *self.output_shape)

    def backward(self, output_gradient, learning_rate):
        batch_size = output_gradient.shape[0]
        return output_gradient.reshape(batch_size, *self.input_shape)