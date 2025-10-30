from math import e
import numpy as np

def tanh(x, beta):
    """
    Description: Takes a real value as an input and returns a value in the range [-1, 1]
    """
    return np.tanh(beta * x)
def sgn(x):
    """
    Description: Takes a real value and returns 1 if its non-negative and -1 otherwise.
    """
    return np.where(x >= 0, 1, -1)