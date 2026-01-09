import numpy as np

class DenseAssociativeMemory:
    def __init__(self, patterns, n=4, beta=5.0, alpha=0.1, lmbda=0.0):
        """
        Description: Initializes a DAM model with the given patterns, energy order, temperature paramter beta,
                     and learning rate alpha.
        =============== Parameters ================
        @param patterns: stored patterns (#patterns x N), where N is the number of neurons in the network.
        @param n: Order of the energy function (use n>2 for dense associative memory).
        @param beta: Tempearture paramter for network updates. Gain for tanh in continuous update.
        @param alpha: Relaxation paramater for updates (0 < alpha <= 1)
        """
        self.patterns = np.array(patterns)
        self.K, self.N = self.patterns.shape # Number of patterns and number of neurons respectively
        self.n = n
        self.beta = beta
        self.alpha = alpha
        self.lmbda = lmbda
    def energy(self, state):
        """Compute DAM energy: - sum_mu F(pattern_mu · state)"""
        projs = self.patterns @ state
        mem_term = -np.sum(self.F(projs))
        reg_term = (self.lmbda/2.0) * np.sum(state**2)


        return mem_term + reg_term
    def F(self, x):
        """
        Description: Rectified polynomial function to be used in energy calculations of the network.
        ============ Parameters ============
        @param x: Input for the function to act upon.
        
        @returns x^n for x > 0; 0 otherwise.
        """
        return np.where(x > 0, x**self.n, 0.0)
    def F_prime(self, x):
        """
        Description: Derivative of the rectified polynomial function F(x).
        ============ Parameters ============
        @param x: Input for the function to act upon.
        
        @returns n*x^(n-1) for x > 0; 0 otherwise.
        """
        # Note: (self.n - 1) can be 0 or positive. 
        # We must handle 0^0 which should be 0 in this context, but 
        # x**0 is 1. np.where handles this correctly.
        return np.where(x > 0, self.n * (x**(self.n - 1)), 0.0)
    def update_neuron(self, state, i):
        """
        Continuous asynchronous update for neuron i.
        """
        s_i = state[i]
        sum_plus = 0.0
        sum_minus = 0.0

        for mu in range(self.K):
            xi = self.patterns[mu, i]
            # projection excluding neuron i (remove current signed contribution)
            proj = np.dot(self.patterns[mu], state) - xi * s_i  

            
            arg_plus  =  xi * s_i + proj   
            arg_minus = -xi * s_i + proj

            sum_plus  += self.F(arg_plus)
            sum_minus += self.F(arg_minus)

        delta = sum_minus - sum_plus

        # continuous relaxation (move toward tanh(beta * delta))
        new_val = np.tanh(self.beta * delta)
        state[i] = (1.0 - self.alpha) * s_i + self.alpha * new_val

        return state
    
    def retrieve(self, noisy_state, steps=500):
        state = noisy_state.copy()
        energy_trace = []
        similarity_trace = []

        for _ in range(steps):
            i = np.random.randint(0, self.N)
            state = self.update_neuron(state, i)
            
            current_iteration_similarity = [np.dot(state, p) / (np.linalg.norm(state) * np.linalg.norm(p) + 1e-12) for p in self.patterns]
            similarity_trace.append(current_iteration_similarity)
            energy_trace.append(self.energy(state))
        
        sims = [np.dot(state, p) / (np.linalg.norm(state) * np.linalg.norm(p) + 1e-12)
                for p in self.patterns]
        best = self.patterns[np.argmax(sims)]

        return state, best, energy_trace, similarity_trace


    def update_neuron_differential(self, state, i):
        """
        Continuous asynchronous update for neuron i based on gradient descent.
        """
        s_i = state[i]
        
        # Calculate the local field h_i = -partial E / partial s_i
        local_field = 0.0
        
        # Calculate all projections at once
        projections = self.patterns @ state # (K,) vector of (P_mu)
        
        # Calculate all F'(P_mu) at once
        F_primes = self.F_prime(projections) # (K,) vector of F'(P_mu)
        
        # Get the i-th column of patterns (all xi_i^mu)
        xi_column = self.patterns[:, i] # (K,) vector of (xi_i^mu)
        
        mem_field = np.dot(xi_column, F_primes)
        reg_field = self.lmbda * s_i
        # h_i = sum_mu (xi_i^mu * F'(P_mu))
        local_field = mem_field - reg_field

        # The target value is the tanh of the local field
        new_val = np.tanh(self.beta * local_field)
        
        # Relaxation update
        state[i] = (1.0 - self.alpha) * s_i + self.alpha * new_val

        return state
    def retrieve_differential(self, noisy_state, steps=500):
        state = noisy_state.copy()
        energy_trace = []
        similarity_trace = []

        for _ in range(steps):
            # Asynchronous update
            i = np.random.randint(0, self.N)
            state = self.update_neuron_differential(state, i)
            
            # Optional: Track metrics every few steps (e.g., every N steps)
            # This can be slow inside the loop
            if _ % 10 == 0:
                current_iteration_similarity = [np.dot(state, p) / (np.linalg.norm(state) * np.linalg.norm(p) + 1e-12) for p in self.patterns]
                similarity_trace.append(current_iteration_similarity)
                energy_trace.append(self.energy(state))
        
        # Final similarity check
        sims = [np.dot(state, p) / (np.linalg.norm(state) * np.linalg.norm(p) + 1e-12)
                for p in self.patterns]
        best = self.patterns[np.argmax(sims)]

        return state, best, energy_trace, similarity_trace