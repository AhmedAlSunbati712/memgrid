from math import e

def tanh(x):
    return (e**(x) - e**(-x))/(e**(x) + e**(-x))
def sgn(x):
    if x == 0: return 0
    elif x > 0: return 1
    else: return -1