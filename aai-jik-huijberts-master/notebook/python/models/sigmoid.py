import math
import numpy as np
class sigmoid():
    def activate(self, hidden_output):
        return 1 / (1+np.exp(-np.array(hidden_output)))
    def activate_math(self, hidden_output):
        out = [ 1/ (1 + math.exp(-x)) for x in hidden_output]
        return np.array(out)
    def differentiate(self, hidden_input):
        x = self.activate(np.array(hidden_input))
        return x + (1-x)
    