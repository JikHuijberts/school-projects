import numpy as np
import math
class softmax():
    def activate(self, hidden_output):
        e_x = np.exp(hidden_output - np.max(hidden_output))
        return e_x / e_x.sum(axis=0)
    def differentiate(self, hidden_input):
        return 
    
