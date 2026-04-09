import numpy as np

def similarity(x, y):
    """
    Description: Calculates cosine similarity between two vectors.
    """
    denom = (np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0:
        return 0
    return np.dot(x, y) / denom