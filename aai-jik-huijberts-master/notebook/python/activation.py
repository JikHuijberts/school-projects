import numpy as np


class Relu:

    def activate(self, hidden_output):
        return np.maximum(0, hidden_output)

    def differentiate(self, hidden_input):
        return 1. * (hidden_input > 0)


class Sigmoid:

    def activate(self, hidden_output):
        return 1 / (1 + np.exp(-np.array(hidden_output)))

    def differentiate(self, hidden_input):
        x = self.activate(np.array(hidden_input))
        return x * (1 - x)


class Softmax:

    def activate(self, hidden_output):
        x = np.exp(hidden_output - np.max(hidden_output))
        return x / x.sum(axis=0)

    def differentiate(self, hidden_input):
        x = self.activate(np.array(hidden_input))
        return x * (1 - x)


class Tanh:

    # Numpy activate as the primary because it's the fastest solution
    def activate(self, hidden_output):
        return np.tanh(hidden_output)

    def differentiate(self, hidden_output):
        x = self.activate(hidden_output)
        return 1 - np.power(x, 2)
