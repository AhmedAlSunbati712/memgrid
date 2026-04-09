import numpy as np
from DAM import SilentDAM
from utils import (
    cosine,
    create_single_scale_encoder,
    encode_points,
    encode_single_point,
    find_nearest_encoded,
    find_nearest_neighbor_2d,
)

def identification_experiment(scales, n_values, K_values, noise_level=0.1,
                               num_trials=5, steps_multiplier=20,
                               n_orientations=3, n_cells=5):
    """
    Test identification accuracy across frequencies and n values.
    
    Identification Task:
    1. Generate K random 2D points and encode them
    2. Store encoded patterns in DAM
    3. For each stored pattern:
       - Add noise to the ORIGINAL 2D point
       - Encode the noisy point
       - Retrieve from DAM
       - Success = retrieved state is closest to the target encoded pattern
    
    Parameters:
    -----------
    scales : list of float
        Spatial frequency scales to test
    n_values : list of int
        Energy function orders to test (2, 3, 4)
    K_values : list of int
        Number of stored patterns to test
    noise_level : float
        Std dev of Gaussian noise added to 2D coordinates
    num_trials : int
        Number of trials per configuration
    steps_multiplier : int
        Number of update steps = steps_multiplier * N
    n_orientations : int
        Number of orientations in grid encoder
    n_cells : int
        Number of cells per orientation
        
    Returns:
    --------
    dict: results[scale][n][K] = {
        'accuracy': float,
        'avg_sim': float,
        'std_sim': float
    }
    """
    results = {}
    
    for scale in scales:
        results[scale] = {}
        encoder = create_single_scale_encoder(scale, n_orientations, n_cells)
        N = 2 * 1 * n_orientations * n_cells  # Output dimension
        
        print(f"\n{'='*70}")
        print(f"SCALE = {scale} (Encoder dimension N = {N})")
        print(f"{'='*70}")
        
        for n in n_values:
            results[scale][n] = {}
            print(f"\n  n = {n}")
            print(f"  {'-'*50}")
            
            for K in K_values:
                successes = []
                similarities = []
                
                for trial in range(num_trials):
                    # Generate K random 2D points in [-1, 1] x [-1, 1]
                    points_2d = np.random.uniform(-1, 1, (K, 2))
                    
                    # Encode all points
                    encoded_patterns = encode_points(encoder, points_2d)
                    
                    # Create DAM with encoded patterns (no regularization)
                    dam = SilentDAM(
                        encoded_patterns,
                        n=n,
                        beta=0.01,  # Lower beta for stability with continuous patterns
                        alpha=0.5,
                        lmbda=0.0   # No regularization as specified
                    )
                    
                    # Test identification for each stored pattern
                    trial_successes = []
                    trial_similarities = []
                    
                    for target_idx in range(K):
                        # Add noise to original 2D point
                        noisy_2d = points_2d[target_idx] + np.random.normal(0, noise_level, 2)
                        noisy_2d = np.clip(noisy_2d, -1, 1)  # Keep in valid range
                        
                        # Encode noisy point
                        noisy_encoded = encode_single_point(encoder, noisy_2d)
                        
                        # Retrieve from DAM
                        retrieved, _, _, _ = dam.retrieve_differential(
                            noisy_encoded,
                            steps=steps_multiplier * N
                        )
                        
                        # Check if retrieved is closest to target
                        retrieved_idx = find_nearest_encoded(retrieved, encoded_patterns)
                        success = (retrieved_idx == target_idx)
                        trial_successes.append(success)
                        
                        # Record similarity to target
                        sim = cosine(retrieved, encoded_patterns[target_idx])
                        trial_similarities.append(sim)
                    
                    successes.extend(trial_successes)
                    similarities.extend(trial_similarities)
                
                accuracy = np.mean(successes) * 100
                avg_sim = np.mean(similarities)
                std_sim = np.std(similarities)
                
                results[scale][n][K] = {
                    'accuracy': accuracy,
                    'avg_sim': avg_sim,
                    'std_sim': std_sim
                }
                
                print(f"    K={K:3d}: Accuracy={accuracy:5.1f}%, "
                      f"Similarity={avg_sim:.4f} ± {std_sim:.4f}")
    
    return results


def generalization_experiment(scales, n_values, K_values, num_test_points=50,
                               num_trials=5, steps_multiplier=20,
                               n_orientations=3, n_cells=5):
    """
    Test generalization accuracy across frequencies and n values.
    
    Generalization Task:
    1. Generate K random 2D points and encode them (stored patterns)
    2. Store encoded patterns in DAM
    3. For M novel test points (NOT in stored set):
       - Encode the novel point
       - Retrieve from DAM
       - Find ground-truth nearest neighbor in original 2D space
       - Success = retrieved state is closest to that neighbor's encoding
    
    Parameters:
    -----------
    scales : list of float
        Spatial frequency scales to test
    n_values : list of int
        Energy function orders to test (2, 3, 4)
    K_values : list of int
        Number of stored patterns to test
    num_test_points : int
        Number of novel test points per trial
    num_trials : int
        Number of trials per configuration
    steps_multiplier : int
        Number of update steps = steps_multiplier * N
    n_orientations : int
        Number of orientations in grid encoder
    n_cells : int
        Number of cells per orientation
        
    Returns:
    --------
    dict: results[scale][n][K] = {
        'accuracy': float,
        'avg_sim': float,
        'std_sim': float
    }
    """
    results = {}
    
    for scale in scales:
        results[scale] = {}
        encoder = create_single_scale_encoder(scale, n_orientations, n_cells)
        N = 2 * 1 * n_orientations * n_cells  # Output dimension
        
        print(f"\n{'='*70}")
        print(f"SCALE = {scale} (Encoder dimension N = {N})")
        print(f"{'='*70}")
        
        for n in n_values:
            results[scale][n] = {}
            print(f"\n  n = {n}")
            print(f"  {'-'*50}")
            
            for K in K_values:
                successes = []
                similarities = []
                
                for trial in range(num_trials):
                    # Generate K random 2D points in [-1, 1] x [-1, 1] (stored)
                    stored_points_2d = np.random.uniform(-1, 1, (K, 2))
                    
                    # Encode stored points
                    encoded_patterns = encode_points(encoder, stored_points_2d)
                    
                    # Create DAM with encoded patterns (no regularization)
                    dam = SilentDAM(
                        encoded_patterns,
                        n=n,
                        beta=0.01,  # Lower beta for stability
                        alpha=0.5,
                        lmbda=0.0   # No regularization
                    )
                    
                    # Generate novel test points
                    test_points_2d = np.random.uniform(-1, 1, (num_test_points, 2))
                    
                    for test_point in test_points_2d:
                        # Find ground-truth nearest neighbor in 2D space
                        gt_nearest_idx = find_nearest_neighbor_2d(test_point, stored_points_2d)
                        
                        # Encode test point
                        test_encoded = encode_single_point(encoder, test_point)
                        
                        # Retrieve from DAM
                        retrieved, _, _, _ = dam.retrieve_differential(
                            test_encoded,
                            steps=steps_multiplier * N
                        )
                        
                        # Check if retrieved is closest to ground-truth nearest neighbor
                        retrieved_idx = find_nearest_encoded(retrieved, encoded_patterns)
                        success = (retrieved_idx == gt_nearest_idx)
                        successes.append(success)
                        
                        # Record similarity to ground-truth nearest neighbor's encoding
                        sim = cosine(retrieved, encoded_patterns[gt_nearest_idx])
                        similarities.append(sim)
                
                accuracy = np.mean(successes) * 100
                avg_sim = np.mean(similarities)
                std_sim = np.std(similarities)
                
                results[scale][n][K] = {
                    'accuracy': accuracy,
                    'avg_sim': avg_sim,
                    'std_sim': std_sim
                }
                
                print(f"    K={K:3d}: Accuracy={accuracy:5.1f}%, "
                      f"Similarity={avg_sim:.4f} ± {std_sim:.4f}")
    
    return results