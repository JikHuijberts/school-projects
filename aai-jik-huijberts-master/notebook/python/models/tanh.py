import math
import numpy as np
class tanh():
    # Numpy activate as the primary because it's the fastest solution
    def activate(hidden_output):
        return np.tanh(hidden_output)

    def activate_hard(hidden_output):
        out = [((math.exp(2* x) -1)/(2*math.exp(-x))) for x in hidden_output]
        return np.array(out)

    def activate_simple_math(hidden_output):
        out = [math.tanh(x) for x in hidden_output]
        return np.array(out)