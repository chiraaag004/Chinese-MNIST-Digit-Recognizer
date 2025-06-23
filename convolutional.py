import numpy as np
from scipy import signal
from layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        # input_shape: (channels, height, width)
        self.input_depth, self.input_height, self.input_width = input_shape
        self.kernel_size = kernel_size
        self.depth = depth

        self.output_height = self.input_height - kernel_size + 1
        self.output_width = self.input_width - kernel_size + 1
        self.output_shape = (depth, self.output_height, self.output_width)
        self.kernels_shape = (depth, self.input_depth, kernel_size, kernel_size)

        self.kernels = np.random.randn(*self.kernels_shape) * 0.1
        self.biases = np.random.randn(*self.output_shape) * 0.1

    def forward(self, input):
        # input shape: (batch_size, channels, height, width)
        self.input = input
        batch_size = input.shape[0]
        self.output = np.zeros((batch_size, *self.output_shape))

        for b in range(batch_size):
            for i in range(self.depth):
                self.output[b, i] = self.biases[i]
                for j in range(self.input_depth):
                    self.output[b, i] += signal.correlate2d(
                        self.input[b, j], self.kernels[i, j], mode="valid"
                    )
        return self.output

    def backward(self, output_gradient, learning_rate):
        # output_gradient shape: (batch_size, depth, out_height, out_width)
        batch_size = output_gradient.shape[0]

        kernels_gradient = np.zeros_like(self.kernels)
        biases_gradient = np.zeros_like(self.biases)
        input_gradient = np.zeros_like(self.input)

        for b in range(batch_size):
            for i in range(self.depth):
                biases_gradient[i] += output_gradient[b, i]
                for j in range(self.input_depth):
                    kernels_gradient[i, j] += signal.correlate2d(
                        self.input[b, j], output_gradient[b, i], mode="valid"
                    )
                    input_gradient[b, j] += signal.convolve2d(
                        output_gradient[b, i], self.kernels[i, j], mode="full"
                    )

        self.kernels -= learning_rate * (kernels_gradient / batch_size)
        self.biases -= learning_rate * (biases_gradient / batch_size)

        return input_gradient
