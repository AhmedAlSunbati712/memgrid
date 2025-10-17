from utils.activation_functions import tanh, sgn
from utils.metrix import similarity
import numpy as np
class Hopfield:
    def __init__(self, activation_function=sgn, n_neurons=10):
        self.activation_function = activation_function
        self.n_neurons = n_neurons
        self.stored_patterns = []
        self.W = np.zeros((self.n_neurons, self.n_neurons))
    def bind(self, X):
        if isinstance(X, np.ndarray) and X.ndim == 2:
            for pattern in X:
                self.bind(pattern)
            return

        # single pattern case
        x = X.reshape(-1)
        self.W += np.outer(x, x)
        np.fill_diagonal(self.W, 0)
        self.stored_patterns.append(x.copy())
    def asynchrous_update(self, state, i):
        flip = self.activation_function(np.dot(self.W[i], state))
        state[i] = flip
        return state
    def synchrous_update(self, state):
        state = self.activation_function(np.dot(self.W, state))
        return state
    def retrieve(self, state, synchrous=True, n_iterations=10):
        state = state.copy()
        for _ in range(n_iterations):
            if synchrous:
                state = self.synchrous_update(state)
            else:
                # choose a random neuron index each time
                i = np.random.randint(0, self.n_neurons)
                state = self.asynchrous_update(state, i)
        
        similarities = [similarity(state, p) for p in self.stored_patterns]
        closest_pattern = self.stored_patterns[np.argmax(similarities)]
        return state, closest_pattern


