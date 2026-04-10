import numpy as np
from numba import njit


@njit(cache=True)
def _numba_retrieve_core(patterns, state, projs, update_indices, n_order, beta, alpha, lmbda):
    """In-place differential retrieval with incremental projection updates."""
    K = patterns.shape[0]
    n_minus_one = n_order - 1

    for t in range(update_indices.shape[0]):
        i = update_indices[t]
        s_i = state[i]

        mem_field = 0.0
        for mu in range(K):
            proj = projs[mu]
            if proj > 0.0:
                fp = n_order * (proj ** n_minus_one)
                mem_field += patterns[mu, i] * fp

        local_field = mem_field - lmbda * s_i
        new_val = np.tanh(beta * local_field)
        s_new = (1.0 - alpha) * s_i + alpha * new_val

        delta = s_new - s_i
        state[i] = s_new
        for mu in range(K):
            projs[mu] += patterns[mu, i] * delta

class DenseAssociativeMemory:
    def __init__(self, patterns, n=4, beta=5.0, alpha=0.1, lmbda=0.0, verbose=False):
        """
        Description: Initializes a DAM model with the given patterns, energy order, temperature paramter beta,
                     and learning rate alpha.
        =============== Parameters ================
        @param patterns: stored patterns (#patterns x N), where N is the number of neurons in the network.
        @param n: Order of the energy function (use n>2 for dense associative memory).
        @param beta: Tempearture paramter for network updates. Gain for tanh in continuous update.
        @param alpha: Relaxation paramater for updates (0 < alpha <= 1)
        """
        self.patterns = np.asarray(patterns, dtype=float)
        self.K, self.N = self.patterns.shape # Number of patterns and number of neurons respectively
        self.n = n
        self.beta = beta
        self.alpha = alpha
        self.lmbda = lmbda
        self.verbose = verbose
        self.pattern_norms = np.linalg.norm(self.patterns, axis=1) + 1e-12
    def energy(self, state):
        """Compute DAM energy: - sum_mu F(pattern_mu · state)"""
        projs = self.patterns @ state
        return self.energy_from_projs(projs, state)

    def energy_from_projs(self, projs, state):
        """Compute DAM energy from precomputed projections."""
        mem_term = -np.sum(self.F(projs))
        reg_term = (self.lmbda/2.0) * np.sum(state**2)
        if self.verbose:
            print("mem_term: ", mem_term)
            print("reg_term: ", reg_term)

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
    def _compute_similarity_trace_row(self, state):
        state_norm = np.linalg.norm(state) + 1e-12
        return (self.patterns @ state) / (self.pattern_norms * state_norm)

    def _prepare_update_indices(self, steps, update_indices):
        if update_indices is None:
            return np.random.randint(0, self.N, size=steps, dtype=np.int64)
        indices = np.asarray(update_indices, dtype=np.int64)
        if indices.shape[0] != steps:
            raise ValueError(
                f"update_indices length ({indices.shape[0]}) must equal steps ({steps})"
            )
        return indices

    def _retrieve_differential_numpy(self, state, steps, update_indices, trace_every):
        energy_trace = []
        similarity_trace = []
        projs = self.patterns @ state

        for t in range(steps):
            i = int(update_indices[t])
            s_i = state[i]

            f_primes = self.F_prime(projs)
            xi_column = self.patterns[:, i]
            mem_field = float(np.dot(xi_column, f_primes))
            local_field = mem_field - self.lmbda * s_i
            new_val = np.tanh(self.beta * local_field)
            s_new = (1.0 - self.alpha) * s_i + self.alpha * new_val

            delta = s_new - s_i
            state[i] = s_new
            projs = projs + xi_column * delta

            if trace_every > 0 and (t % trace_every == 0):
                similarity_trace.append(self._compute_similarity_trace_row(state).tolist())
                energy_trace.append(float(self.energy_from_projs(projs, state)))

        sims = self._compute_similarity_trace_row(state)
        best_idx = int(np.argmax(sims))
        best = self.patterns[best_idx]
        return state, best, energy_trace, similarity_trace, best_idx

    def _retrieve_differential_numba(self, state, steps, update_indices, trace_every):
        if trace_every > 0:
            raise ValueError("backend='numba' currently requires trace_every=0")

        state = np.asarray(state, dtype=np.float64).copy()
        patterns = np.ascontiguousarray(self.patterns, dtype=np.float64)
        projs = patterns @ state
        _numba_retrieve_core(
            patterns,
            state,
            projs,
            np.ascontiguousarray(update_indices, dtype=np.int64),
            int(self.n),
            float(self.beta),
            float(self.alpha),
            float(self.lmbda),
        )
        sims = self._compute_similarity_trace_row(state)
        best_idx = int(np.argmax(sims))
        best = self.patterns[best_idx]
        return state, best, [], [], best_idx

    def retrieve_differential(
        self,
        noisy_state,
        steps=500,
        update_indices=None,
        trace_every=10,
        backend="numpy",
        return_best_idx=False,
    ):
        state = np.asarray(noisy_state, dtype=float).copy()
        update_indices = self._prepare_update_indices(steps, update_indices)

        if backend == "numpy":
            result = self._retrieve_differential_numpy(
                state, steps, update_indices, trace_every
            )
        elif backend == "numba":
            result = self._retrieve_differential_numba(
                state, steps, update_indices, trace_every
            )
        else:
            raise ValueError(f"Unknown backend '{backend}'. Expected 'numpy' or 'numba'.")

        if return_best_idx:
            return result
        return result[:4]


class SilentDAM(DenseAssociativeMemory):
    """
    Backward-compatible alias used by experiments to ensure quiet energy calls.
    """
    def __init__(self, patterns, n=4, beta=5.0, alpha=0.1, lmbda=0.0):
        super().__init__(patterns, n=n, beta=beta, alpha=alpha, lmbda=lmbda, verbose=False)