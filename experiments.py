import numpy as np
from Encoder import GridEncoder
from DAM import SilentDAM
from utils import (
    N_CELLS_PER_ORI,
    N_ORIENTATIONS,
    BASE_FREQ,
    cosine,
    create_multiscale_encoder,
    create_single_scale_encoder,
    encode_points,
    encode_single_point,
    find_nearest_encoded,
    find_nearest_neighbor_2d,
)


def _multiscale_grid_encoder(scale_factor, n_modules, n_orientations, n_cells_per_orientation, base_freq):
    """
    GridEncoder with geometric module frequencies base_freq * scale_factor**m.
    Uses utils.create_multiscale_encoder when layout matches its fixed orientations/cells.
    """
    if (
        n_orientations == N_ORIENTATIONS
        and n_cells_per_orientation == N_CELLS_PER_ORI
    ):
        return create_multiscale_encoder(scale_factor, n_modules=n_modules, base_freq=base_freq)
    scales = base_freq * (scale_factor ** np.arange(n_modules))
    return GridEncoder(
        n_modules=n_modules,
        n_orientations=n_orientations,
        n_cells_per_orientation=n_cells_per_orientation,
        scales=scales,
    )


def _multiscale_encoding_dim(n_modules, n_orientations, n_cells_per_orientation):
    return 2 * n_modules * n_orientations * n_cells_per_orientation

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


def identification_experiment_multiscale(
    scale_factors,
    n_values,
    K_values,
    noise_level=0.1,
    num_trials=5,
    steps_multiplier=20,
    n_modules=4,
    n_orientations=3,
    n_cells_per_orientation=4,
    base_freq=0.5,
    beta=0.01,
    alpha=0.5,
    lmbda=0.0,
    verbose=False,
):
    """
    Identification with multi-scale grid codes (noise in encoded space; see notebook).

    Returns dict results[c][n][K] with keys accuracy, avg_sim, std_sim.
    """
    results = {}
    enc_dim = _multiscale_encoding_dim(
        n_modules, n_orientations, n_cells_per_orientation
    )

    for c in scale_factors:
        results[c] = {}
        encoder = _multiscale_grid_encoder(
            c, n_modules, n_orientations, n_cells_per_orientation, base_freq
        )
        freqs = base_freq * (c ** np.arange(n_modules))

        if verbose:
            print(f"\n{'='*70}")
            print(f"SCALE FACTOR c = {c} (frequencies: {freqs})")
            print(f"{'='*70}")

        for n in n_values:
            results[c][n] = {}
            if verbose:
                print(f"\n  n = {n}")
                print(f"  {'-'*50}")

            for K in K_values:
                successes = []
                similarities = []

                for _trial in range(num_trials):
                    points_2d = np.random.uniform(-1, 1, (K, 2))
                    patterns_encoded = encode_points(encoder, points_2d)

                    dam = SilentDAM(
                        patterns_encoded,
                        n=n,
                        beta=beta,
                        alpha=alpha,
                        lmbda=lmbda,
                    )

                    for target_idx in range(K):
                        target_encoded = patterns_encoded[target_idx]
                        noisy_query = target_encoded + np.random.normal(
                            0, noise_level, enc_dim
                        )
                        retrieved, _, _, _ = dam.retrieve_differential(
                            noisy_query,
                            steps=steps_multiplier * enc_dim,
                        )
                        retrieved_idx = find_nearest_encoded(
                            retrieved, patterns_encoded
                        )
                        successes.append(retrieved_idx == target_idx)
                        similarities.append(
                            cosine(retrieved, target_encoded)
                        )

                accuracy = np.mean(successes) * 100
                avg_sim = np.mean(similarities)
                std_sim = np.std(similarities)
                results[c][n][K] = {
                    "accuracy": accuracy,
                    "avg_sim": avg_sim,
                    "std_sim": std_sim,
                }
                if verbose:
                    print(
                        f"    K={K:3d}: Accuracy={accuracy:5.1f}%, "
                        f"Similarity={avg_sim:.4f} ± {std_sim:.4f}"
                    )

    return results


def generalization_experiment_multiscale(
    scale_factors,
    n_values,
    K_values,
    num_test_points=50,
    num_trials=5,
    steps_multiplier=20,
    n_modules=4,
    n_orientations=3,
    n_cells_per_orientation=4,
    base_freq=0.5,
    beta=0.01,
    alpha=0.5,
    lmbda=0.0,
    sample_2d_fn=None,
    verbose=False,
):
    """
    Generalization with multi-scale grid codes.

    If sample_2d_fn is None, points are drawn from Uniform([-1,1]^2). Otherwise
    sample_2d_fn(shape) must return an ndarray of the given shape (last axis 2).

    Returns dict results[c][n][K] with keys accuracy, avg_sim, std_sim.
    """
    results = {}
    enc_dim = _multiscale_encoding_dim(
        n_modules, n_orientations, n_cells_per_orientation
    )

    def sample_points(shape):
        if sample_2d_fn is None:
            return np.random.uniform(-1, 1, shape)
        try:
            points = np.asarray(sample_2d_fn(shape))
        except TypeError:
            points = np.asarray(sample_2d_fn(shape[0]))
        return points.reshape(shape)

    for c in scale_factors:
        results[c] = {}
        encoder = _multiscale_grid_encoder(
            c, n_modules, n_orientations, n_cells_per_orientation, base_freq
        )
        freqs = base_freq * (c ** np.arange(n_modules))

        if verbose:
            print(f"\n{'='*70}")
            print(f"SCALE FACTOR c = {c} (frequencies: {freqs})")
            print(f"{'='*70}")

        for n in n_values:
            results[c][n] = {}
            if verbose:
                print(f"\n  n = {n}")
                print(f"  {'-'*50}")

            for K in K_values:
                successes = []
                similarities = []

                for _trial in range(num_trials):
                    points_2d = sample_points((K, 2))
                    patterns_encoded = encode_points(encoder, points_2d)

                    dam = SilentDAM(
                        patterns_encoded,
                        n=n,
                        beta=beta,
                        alpha=alpha,
                        lmbda=lmbda,
                    )

                    test_points_2d = sample_points((num_test_points, 2))
                    for test_point in test_points_2d:
                        gt_nearest_idx = find_nearest_neighbor_2d(
                            test_point, points_2d
                        )
                        test_encoded = encode_single_point(encoder, test_point)
                        retrieved, _, _, _ = dam.retrieve_differential(
                            test_encoded,
                            steps=steps_multiplier * enc_dim,
                        )
                        retrieved_idx = find_nearest_encoded(
                            retrieved, patterns_encoded
                        )
                        successes.append(retrieved_idx == gt_nearest_idx)
                        similarities.append(
                            cosine(
                                retrieved, patterns_encoded[gt_nearest_idx]
                            )
                        )

                accuracy = np.mean(successes) * 100
                avg_sim = np.mean(similarities)
                std_sim = np.std(similarities)
                results[c][n][K] = {
                    "accuracy": accuracy,
                    "avg_sim": avg_sim,
                    "std_sim": std_sim,
                }
                if verbose:
                    print(
                        f"    K={K:3d}: Accuracy={accuracy:5.1f}%, "
                        f"Similarity={avg_sim:.4f} ± {std_sim:.4f}"
                    )

    return results


def run_multiscale_ident_gen_sweep(
    scale_factors,
    n_values,
    K_values,
    noise_level=0.1,
    num_test_points=50,
    num_trials=5,
    steps_multiplier=20,
    n_modules=4,
    n_orientations=3,
    n_cells_per_orientation=4,
    base_freq=0.5,
    beta=0.01,
    alpha=0.5,
    lmbda=0.0,
    sample_2d_fn=None,
    verbose=False,
):
    """
    Run identification and generalization multiscale experiments with shared hyperparameters.

    Returns
    -------
    tuple
        (ident_results, gen_results) each mapping c -> n -> K -> dict with
        accuracy, avg_sim, std_sim.
    """
    ident_results = identification_experiment_multiscale(
        scale_factors,
        n_values,
        K_values,
        noise_level=noise_level,
        num_trials=num_trials,
        steps_multiplier=steps_multiplier,
        n_modules=n_modules,
        n_orientations=n_orientations,
        n_cells_per_orientation=n_cells_per_orientation,
        base_freq=base_freq,
        beta=beta,
        alpha=alpha,
        lmbda=lmbda,
        verbose=verbose,
    )
    gen_results = generalization_experiment_multiscale(
        scale_factors,
        n_values,
        K_values,
        num_test_points=num_test_points,
        num_trials=num_trials,
        steps_multiplier=steps_multiplier,
        n_modules=n_modules,
        n_orientations=n_orientations,
        n_cells_per_orientation=n_cells_per_orientation,
        base_freq=base_freq,
        beta=beta,
        alpha=alpha,
        lmbda=lmbda,
        sample_2d_fn=sample_2d_fn,
        verbose=verbose,
    )
    return ident_results, gen_results


def structure_preservation_analysis(
    scale_factors,
    n_points=100,
    n_samples=500,
    n_modules=4,
    n_orientations=3,
    n_cells_per_orientation=4,
    base_freq=0.5,
):
    """
    Analyze preservation of 2D distance structure under multiscale encodings.

    Returns
    -------
    dict
        results[c] = {"correlation": float, "p_value": float}
    """
    points_2d = np.random.uniform(-1, 1, (n_points, 2))
    pair_indices = np.array(
        [(i, j) for i in range(n_points) for j in range(i + 1, n_points)]
    )
    if len(pair_indices) > n_samples:
        sampled_idx = np.random.choice(len(pair_indices), n_samples, replace=False)
        pair_indices = pair_indices[sampled_idx]

    results = {}
    for c in scale_factors:
        encoder = _multiscale_grid_encoder(
            c, n_modules, n_orientations, n_cells_per_orientation, base_freq
        )
        encoded_points = encode_points(encoder, points_2d)

        d_2d = []
        d_encoded = []
        for i, j in pair_indices:
            d_2d.append(np.linalg.norm(points_2d[i] - points_2d[j]))
            d_encoded.append(1 - cosine(encoded_points[i], encoded_points[j]))

        d_2d = np.array(d_2d)
        d_encoded = np.array(d_encoded)
        corr_matrix = np.corrcoef(d_2d, d_encoded)
        correlation = float(corr_matrix[0, 1])
        # Keep p_value key for notebook compatibility without scipy dependency.
        results[c] = {"correlation": correlation, "p_value": np.nan}
    return results


def get_generalization_optimum_c(
    gen_results,
    ident_results,
    scale_factors,
    n_values,
    K_values,
    n_for_opt=4,
):
    """
    Return c that maximizes mean generalization at n=n_for_opt.
    """
    if n_for_opt not in n_values:
        raise ValueError(f"n_for_opt={n_for_opt} not found in n_values={n_values}")

    best_c = None
    best_gen = -np.inf
    for c in scale_factors:
        mean_gen = float(np.mean([gen_results[c][n_for_opt][K]["accuracy"] for K in K_values]))
        if mean_gen > best_gen:
            best_gen = mean_gen
            best_c = c
    mean_ident = float(np.mean([ident_results[best_c][n_for_opt][K]["accuracy"] for K in K_values]))
    return best_c, best_gen, mean_ident


def analyze_multiscale_tradeoff(
    ident_results,
    gen_results,
    scale_factors,
    n_values,
    K_values,
    print_summary=True,
):
    """
    Aggregate multiscale tradeoff summaries and optionally print compact tables.

    Returns
    -------
    dict
        {
          "rows": [...],
          "trends": [...]
        }
    """
    rows = []
    for n in n_values:
        for c in scale_factors:
            freqs = BASE_FREQ * (float(c) ** np.arange(4))
            for K in K_values:
                ident_acc = float(ident_results[c][n][K]["accuracy"])
                gen_acc = float(gen_results[c][n][K]["accuracy"])
                diff = ident_acc - gen_acc
                if diff > 10:
                    label = "Identification > Generalization"
                elif diff < -10:
                    label = "Generalization > Identification"
                else:
                    label = "Balanced"
                rows.append(
                    {
                        "n": int(n),
                        "scale_c": float(c),
                        "frequencies": freqs.tolist(),
                        "K": int(K),
                        "ident_accuracy": ident_acc,
                        "gen_accuracy": gen_acc,
                        "diff_ident_minus_gen": diff,
                        "tradeoff_label": label,
                    }
                )

    trends = []
    if len(scale_factors) >= 2:
        small_c = scale_factors[0]
        large_c = scale_factors[-1]
        for n in n_values:
            small_ident = float(np.mean([ident_results[small_c][n][K]["accuracy"] for K in K_values]))
            large_ident = float(np.mean([ident_results[large_c][n][K]["accuracy"] for K in K_values]))
            small_gen = float(np.mean([gen_results[small_c][n][K]["accuracy"] for K in K_values]))
            large_gen = float(np.mean([gen_results[large_c][n][K]["accuracy"] for K in K_values]))
            ident_trend = "increases" if large_ident > small_ident else "decreases"
            gen_trend = "increases" if large_gen > small_gen else "decreases"
            trends.append(
                {
                    "n": int(n),
                    "small_c": float(small_c),
                    "large_c": float(large_c),
                    "ident_trend": ident_trend,
                    "gen_trend": gen_trend,
                }
            )

    if print_summary:
        print("=" * 90)
        print("SUMMARY: MULTI-SCALE GRID CODES - IDENTIFICATION VS GENERALIZATION")
        print("=" * 90)
        for n in n_values:
            print(f"\nn = {n}")
            print("-" * 90)
            for row in [r for r in rows if r["n"] == n]:
                print(
                    f"c={row['scale_c']:.2f}, K={row['K']:>3} | "
                    f"Ident={row['ident_accuracy']:>6.1f}% | "
                    f"Gen={row['gen_accuracy']:>6.1f}% | "
                    f"Δ={row['diff_ident_minus_gen']:>6.1f}% | "
                    f"{row['tradeoff_label']}"
                )
        if trends:
            print("\nTREND CHECK (small c -> large c)")
            for trend in trends:
                print(
                    f"n={trend['n']}: identification {trend['ident_trend']}, "
                    f"generalization {trend['gen_trend']}"
                )

    return {"rows": rows, "trends": trends}


def run_breakit_sweep(
    scale_factors,
    n_values,
    K_values,
    n_modules=4,
    n_orientations=3,
    n_cells_per_orientation=4,
    base_freq=0.5,
    noise_level=0.1,
    num_trials=3,
    num_test_points=50,
    steps_multiplier=20,
    sample_2d_fn=None,
    verbose=False,
):
    """
    Backward-compatible notebook API for Part 4b break-it sweeps.
    """
    return run_multiscale_ident_gen_sweep(
        scale_factors=scale_factors,
        n_values=n_values,
        K_values=K_values,
        noise_level=noise_level,
        num_test_points=num_test_points,
        num_trials=num_trials,
        steps_multiplier=steps_multiplier,
        n_modules=n_modules,
        n_orientations=n_orientations,
        n_cells_per_orientation=n_cells_per_orientation,
        base_freq=base_freq,
        sample_2d_fn=sample_2d_fn,
        verbose=verbose,
    )


def sample_uniform(K):
    return np.random.uniform(-1, 1, (K, 2))


def sample_clustered(K, n_clusters=3, sigma=0.2):
    centers = np.random.uniform(-0.7, 0.7, (n_clusters, 2))
    idx = np.random.randint(0, n_clusters, size=K)
    return centers[idx] + np.random.normal(0, sigma, (K, 2))


def sample_biased(K):
    return np.random.uniform(0, 1, (K, 2))