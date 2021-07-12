import numpy as np
from numpy.core.fromnumeric import size


class NeuralNetwork():

    def __init__(self, layer_sizes):

        # TODO
        # layer_sizes example: [4, 10, 2]
        # weight for the first layer
        self.weight_first = np.random.normal(size=(layer_sizes[1] , layer_sizes[0]))
        # weight for the second layer
        self.weight_second = np.random.normal(size=(layer_sizes[2] , layer_sizes[1]))
        # bias of the first layer
        self.bias_first = np.zeros((layer_sizes[1], 1))
        # bias of the second layer
        self.bias_second = np.zeros((layer_sizes[2], 1))
    
    def activation(self, x):
        
        # TODO sigmoid activation function
        return 1/(1 + np.exp(-x))

    def forward(self, x):
        
        # TODO
        # x example: np.array([[0.1], [0.2], [0.3]])
        hidden_layer = self.activation(self.weight_first @ x + self.bias_first)
        output_layer = self.activation(self.weight_second @ hidden_layer + self.bias_second)
        
        return output_layer
