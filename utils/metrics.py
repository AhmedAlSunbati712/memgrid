import numpy as np

def similarity(x, y):
    np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))