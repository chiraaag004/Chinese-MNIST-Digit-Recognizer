from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__ (self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(self.input, self.weights) + self.bias
    

    def backward(self, output_gradient, learning_rate):

        batch_size = output_gradient.shape[0]

        weights_gradient = np.dot(self.input.T, output_gradient) / batch_size
        bias_gradient = np.mean(output_gradient, axis=0, keepdims=True)

        # np.clip(weights_gradient, -1, 1, out=weights_gradient)
        # np.clip(bias_gradient, -1, 1, out=bias_gradient)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return np.dot(output_gradient, self.weights.T)
