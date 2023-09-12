import numpy as np
import math
from loss import mse


class Model:

    def __init__(self, lr=0.01, lf=mse):
        self.layers = []
        self.lr = lr
        self.lf = lf

    def fit(self, x, y, epochs=5):
        fit = []
        # Predict stores al intermediate results in the layer.
        for epoch in range(epochs):

            o = self.predict(x)
            # y = np.array(y, ndmin=2)
            e = y - o
            # Loss function
            loss = self.lf(self.predict(x), y)
            fit.append(loss)
            # targets y - final outputs
            for layer in reversed(self.layers):
                self.update_layer(layer, e, self.lr)
                # calculate hidden errors
                e = np.dot(e, layer.weights.T)
        return fit

    def predict(self, values):
        result = np.array(values, ndmin=2)

        for layer in self.layers:
            result = layer.feed_forward(result)
        return result

    def update_layer(self, layer, errors, lr):
        delta_weights = lr * np.dot(layer.clear_neurons.T,errors * layer.activation.differentiate(layer.output_neurons))
        layer.weights = np.add(layer.weights, delta_weights)
        delta_biases = lr * errors * layer.activation.differentiate(layer.output_neurons)
        layer.biases = np.add(layer.biases, delta_biases)

    def add(self, layer):
        self.layers.append(layer)


class Layer:
    #     return [[], weights, biases, activation, []]

    def __init__(self, input_neurons, neurons, activation=None):
        self.clear_neurons = []
        self.weights = 0.10 * np.random.randn(input_neurons, neurons)
        # self.weights = np.random.normal(loc=0.0, scale=np.sqrt(2 / (input_neurons + neurons)),
        #                          size=(neurons, input_neurons))
        self.activation = activation
        self.biases = np.zeros((1, neurons))
        self.neurons = neurons
        self.output_neurons = []

    def feed_forward(self, values):
        # Bewaar input en resultaten in de laag voor later gebruik.
        self.clear_neurons = values

        self.output_neurons = self.activation.activate(np.dot(values, self.weights) + self.biases)
        return self.output_neurons
