from utils.activation_functions import tanh, sgn
from utils.metrics import similarity
import numpy as np
class Hopfield:
    """
    Description: Implements a classical Hopfield network for storing and retrieving patterns
                 using Hebbian learning and iterative neuron updates.
    """
    def __init__(self, activation_function=sgn, n_neurons=10):
        """
        Description: Initializes a Hopfield network with a specified number of neurons and an activation function.
                     The weight matrix is initialized to zeros, and no patterns are stored initially.

        ========= Parameters =========
        @param activation_function: The activation function used for neuron updates (default is sgn for binary states).
        @param n_neurons: Number of neurons in the network.

        ========= Returns =========
        @return None
        """
        self.activation_function = activation_function
        self.n_neurons = n_neurons
        self.stored_patterns = []
        self.W = np.zeros((self.n_neurons, self.n_neurons))
    def bind(self, X):
        """
        Description: Stores a pattern (or multiple patterns) in the Hopfield network using Hebbian learning.
                     Updates the weight matrix according to the outer product of each pattern with itself,
                     and zeroes out the diagonal to avoid self-connections.

        ========= Parameters =========
        @param X: Single pattern as a 1D array, or multiple patterns as a 2D array (shape: num_patterns x n_neurons).

        ========= Returns =========
        @return None
        """
        if isinstance(X, np.ndarray) and X.ndim == 2:
            for pattern in X:
                self.bind(pattern)
            return

        x = X.reshape(-1)
        self.W += np.outer(x, x)
        np.fill_diagonal(self.W, 0)
        self.stored_patterns.append(x.copy())
    def asynchrous_update(self, state, i):
        """
        Description: Updates a single neuron asynchronously based on the current state and the weight matrix.
                     The specified neuron index is updated using the activation function.

        ========= Parameters =========
        @param state: Current state of the network (1D array of length n_neurons).
        @param i: Index of the neuron to update.

        ========= Returns =========
        @return state: Updated state after flipping neuron i.
        """
        flip = self.activation_function(np.dot(self.W[i], state))
        state[i] = flip
        return state
    def synchrous_update(self, state):
        """
        Description: Updates all neurons simultaneously (synchronous update) based on the current state
                     and the weight matrix, applying the activation function to the resulting vector.

        ========= Parameters =========
        @param state: Current state of the network (1D array of length n_neurons).

        ========= Returns =========
        @return state: Updated state after synchronous update.
        """
        state = self.activation_function(np.dot(self.W, state))
        return state
    def retrieve(self, state, synchrous=True, n_iterations=10):
        """
        Description: Retrieves a stored pattern from the network starting from an initial state.
                     Can perform either synchronous or asynchronous updates for a specified number of iterations.
                     Returns the final state and the stored pattern most similar to it.

        ========= Parameters =========
        @param state: Initial state of the network (1D array of length n_neurons).
        @param synchrous: Boolean flag indicating whether to use synchronous (True) or asynchronous (False) updates.
        @param n_iterations: Number of update iterations to perform.

        ========= Returns =========
        @return state: Final network state after updates.
        @return closest_pattern: Stored pattern most similar to the final state.
        """
        state = state.copy()
        for _ in range(n_iterations):
            if synchrous:
                state = self.synchrous_update(state)
            else:
                # choose a random neuron index each time
                i = np.random.randint(0, self.n_neurons)
                state = self.asynchrous_update(state, i)
        
        similarities = [similarity(state, p) for p in self.stored_patterns]
        print(similarities)
        closest_pattern = self.stored_patterns[np.argmax(similarities)]
        return state, closest_pattern


