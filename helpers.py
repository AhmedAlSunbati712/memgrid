import numpy as np
from datasets import load_dataset


def add_noise_classical(state, noise_level=0.25):
    """
    Description: Given a classical Hopfield network state (each neuron is either -1 or 1), this function
                 chooses randomly (noise_level * n_neurons) neurons to flip their sign.

    ================= Parameters =================
    @param state (numpy.ndarray): A vector that represents the state of the network. Takes shape (n_neurons, ).
    @param noise_level (int): The amount of noise that we want to introduce into that vector.

    ================= Return =================
    @return noisy_state (numpy.ndarray): The given state vector after introducing noise.
    """
    noisy_state = state.copy()
    n_flips = int(noise_level * len(state))
    chosen_indices = np.random.choice(len(state), n_flips, replace=False)
    noisy_state[chosen_indices] *= -1
    return noisy_state

def load_fashion_dataset():
    """
    Description: Loads the Fashion-MNIST dataset and returns the training and test splits.
                 The dataset is shuffled using a fixed random seed for reproducibility.
    ================= Return =================
    @return X_train (Dataset): The training split of the Fashion-MNIST dataset.
    @return X_test (Dataset): The test split of the Fashion-MNIST dataset.
    """
    X_train, X_test = load_dataset("fashion_mnist", split=["train", "test"])
    return X_train.shuffle(seed=999), X_test.shuffle(seed=999)


def binarize_image(X):
    """
    Description: Converts grayscale image(s) into binary Hopfield network state(s).
                 Each pixel is binarized such that white (1) becomes +1 and black (0) becomes -1.
                 If a list of images is provided, the function applies this process to each image.

    ================= Parameters =================
    @param X (PIL.Image or list): A single grayscale image or a list of images to be binarized.

    ================= Return =================
    @return state (numpy.ndarray or list): A flattened 1D binary vector (values in {-1, +1})
                                           representing the image, or a list of such vectors
                                           if the input was a list of images.
    """
    if isinstance(X, list):
        return [binarize_image(state) for state in X]
    
    pattern = X
    state = np.asarray(pattern.convert("1"), dtype=int)
    # map 0 → -1, 1 → +1
    state = state * 2 - 1
    return state