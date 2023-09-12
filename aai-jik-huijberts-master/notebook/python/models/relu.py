import math
import numpy as np
class Relu:
    def activate(self,hidden_output):
        out = [max(0.0, x) for x in hidden_output]
        return np.array(out)
    def differentiate(self, hidden_input):
        return 1. * (hidden_input > 0) 