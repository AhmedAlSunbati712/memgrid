import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from Encoder import GridEncoder
# Utility functions
# ========================= Metrics =========================
def cosine(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

# ========================= Encoders Helper Functions =========================
N_MODULES = 4 
BASE_FREQ = 0.5
N_ORIENTATIONS = 3
N_CELLS_PER_ORI = 4
def create_multiscale_encoder(
    scale_factor,
    n_modules=N_MODULES,
    base_freq=BASE_FREQ,
    n_orientations=N_ORIENTATIONS,
    n_cells_per_orientation=N_CELLS_PER_ORI,
):
    """
    Create GridEncoder with specified scale factor c.
    
    Frequencies follow geometric progression: [q, c*q, c²*q, c³*q, ...]
    
    Parameters:
    -----------
    scale_factor : float
        The scaling ratio c between adjacent modules
    n_modules : int
        Number of grid modules
    base_freq : float
        Base frequency q for the first module
        
    Returns:
    --------
    GridEncoder with configured scales
    """
    scales = base_freq * (scale_factor ** np.arange(n_modules))
    return GridEncoder(
        n_modules=n_modules,
        n_orientations=n_orientations,
        n_cells_per_orientation=n_cells_per_orientation,
        scales=scales,
    )

def create_single_scale_encoder(scale, n_orientations=3, n_cells=5):
    """
    Create a GridEncoder with a single spatial frequency scale.
    
    Parameters:
    scale : float
        Spatial frequency scale. Lower = coarser grid, Higher = finer grid
    n_orientations : int
        Number of orientation directions (default 3: 0°, 60°, 120°)
    n_cells : int
        Number of grid cells per orientation
        
    Returns:
    GridEncoder with output dimension = 2 * n_orientations * n_cells
    """
    return GridEncoder(
        n_modules=1,  # Single frequency
        n_orientations=n_orientations,
        n_cells_per_orientation=n_cells,
        scales=[scale]
    )

def encode_single_point(encoder, point_2d):
    """
    Encode a single 2D point using the grid encoder.
    Works around the axis issue in the original encoder.
    
    Parameters:
    encoder : GridEncoder
        The grid encoder instance
    point_2d : ndarray of shape (2,) or (1, 2)
        Single 2D point
        
    Returns:
    ndarray of shape (N,) where N is the encoder output dimension
    """
    # Delegate to the encoder to keep one canonical implementation.
    return np.asarray(encoder.encode(np.asarray(point_2d).reshape(2,)))

def encode_points(encoder, points_2d):
    """
    Encode multiple 2D points using the grid encoder.
    
    Parameters:
    encoder : GridEncoder
        The grid encoder instance
    points_2d : ndarray of shape (K, 2)
        K points in 2D space
        
    Returns:
    ndarray of shape (K, N) where N is the encoder output dimension
    """
    return np.asarray(encoder.encode(np.asarray(points_2d)))

def find_nearest_neighbor_2d(query_2d, stored_2d_points):
    """
    Find the index of the nearest neighbor in 2D Euclidean space.
    
    Parameters:
    query_2d : ndarray of shape (2,) or (1, 2)
        Query point in 2D
    stored_2d_points : ndarray of shape (K, 2)
        K stored points in 2D
        
    Returns:
    int : Index of the nearest stored point
    """
    query = query_2d.flatten()
    distances = np.linalg.norm(stored_2d_points - query, axis=1)
    return np.argmin(distances)

def find_nearest_cosine(query, stored_patterns):
    """Find index of pattern with highest cosine similarity to query."""
    similarities = np.array([cosine(query, p) for p in stored_patterns])
    return np.argmax(similarities)
    
def find_nearest_euclidean_2d(query_2d, stored_2d):
    """Find index of nearest pattern in 2D space (Euclidean distance)."""
    return find_nearest_neighbor_2d(query_2d, stored_2d)


def find_nearest_encoded(query_encoded, stored_encoded):
    """
    Find the index of the nearest pattern by cosine similarity.
    
    Parameters:
    query_encoded : ndarray of shape (N,) or (1, N)
        Query pattern in encoded space
    stored_encoded : ndarray of shape (K, N)
        K stored patterns
        
    Returns:
    int : Index of the pattern with highest cosine similarity
    """
    query = query_encoded.flatten()
    similarities = np.array([cosine(query, p) for p in stored_encoded])
    return np.argmax(similarities)

# =========================== Analysis Helper Functions ===========================
def analyze_similarity_structure(scales, n_orientations=3, n_cells=5, num_points=100):
    """
    Analyze how well different frequency scales preserve similarity structure.
    
    Measures the correlation between:
    - Pairwise distances in 2D input space (Euclidean)
    - Pairwise distances in encoded space (cosine distance = 1 - cosine similarity)
    
    Higher correlation = better structure preservation = less compression
    Lower correlation = worse structure preservation = more compression
    """
    
    # Generate random 2D points
    points_2d = np.random.uniform(-1, 1, (num_points, 2))
    
    # Compute pairwise 2D distances
    distances_2d = []
    for i in range(num_points):
        for j in range(i+1, num_points):
            d = np.linalg.norm(points_2d[i] - points_2d[j])
            distances_2d.append(d)
    distances_2d = np.array(distances_2d)
    
    results = {}
    
    print("="*70)
    print("SIMILARITY STRUCTURE ANALYSIS")
    print("Correlation between 2D distances and encoded distances")
    print("="*70)
    
    for scale in scales:
        encoder = create_single_scale_encoder(scale, n_orientations, n_cells)
        encoded = encode_points(encoder, points_2d)
        
        # Compute pairwise cosine distances in encoded space
        distances_encoded = []
        for i in range(num_points):
            for j in range(i+1, num_points):
                sim = cosine(encoded[i], encoded[j])
                d = 1 - sim  # Cosine distance
                distances_encoded.append(d)
        distances_encoded = np.array(distances_encoded)
        
        # Compute correlation
        correlation = np.corrcoef(distances_2d, distances_encoded)[0, 1]
        
        results[scale] = {
            'correlation': correlation,
            'distances_2d': distances_2d,
            'distances_encoded': distances_encoded
        }
        
        print(f"  Scale={scale}: Correlation = {correlation:.4f}")
    
    return results

# =========================== Visualization Helpers ===========================

def plot_tradeoff_curves(ident_results, gen_results, scales, n_values, K_values):
    """
    Plot identification vs generalization accuracy to visualize the tradeoff.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(scales)))
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, n in enumerate(n_values):
        ax = axes[idx]
        
        for scale_idx, scale in enumerate(scales):
            ident_accs = []
            gen_accs = []
            
            for K in K_values:
                ident_accs.append(ident_results[scale][n][K]['accuracy'])
                gen_accs.append(gen_results[scale][n][K]['accuracy'])
            
            # Plot with K values as markers along the line
            ax.plot(gen_accs, ident_accs, 
                   color=colors[scale_idx], 
                   marker=markers[scale_idx],
                   markersize=10,
                   linewidth=2,
                   label=f'Scale={scale}')
            
            # Annotate with K values
            for i, K in enumerate(K_values):
                ax.annotate(f'K={K}', (gen_accs[i], ident_accs[i]),
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Generalization Accuracy (%)', fontsize=12)
        ax.set_ylabel('Identification Accuracy (%)', fontsize=12)
        ax.set_title(f'n = {n}', fontsize=14)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 105])
        ax.set_ylim([0, 105])
        
        # Add diagonal reference line (no tradeoff)
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='_nolegend_')
    
    plt.suptitle('Identification vs Generalization Tradeoff\n(Each point is a different K value)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_frequency_effects(ident_results, gen_results, scales, n_values, K_values):
    """
    Plot how scale (frequency) affects identification and generalization separately.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(K_values)))
    
    for idx, n in enumerate(n_values):
        # Top row: Identification
        ax_ident = axes[0, idx]
        # Bottom row: Generalization  
        ax_gen = axes[1, idx]
        
        for k_idx, K in enumerate(K_values):
            ident_accs = [ident_results[s][n][K]['accuracy'] for s in scales]
            gen_accs = [gen_results[s][n][K]['accuracy'] for s in scales]
            
            ax_ident.plot(scales, ident_accs, 
                         color=colors[k_idx],
                         marker='o', markersize=8, linewidth=2,
                         label=f'K={K}')
            ax_gen.plot(scales, gen_accs,
                       color=colors[k_idx],
                       marker='s', markersize=8, linewidth=2,
                       label=f'K={K}')
        
        ax_ident.set_xlabel('Scale (Frequency)', fontsize=11)
        ax_ident.set_ylabel('Identification Accuracy (%)', fontsize=11)
        ax_ident.set_title(f'n = {n}', fontsize=13)
        ax_ident.legend(loc='best', fontsize=9)
        ax_ident.grid(True, alpha=0.3)
        ax_ident.set_xscale('log')
        ax_ident.set_ylim([0, 105])
        
        ax_gen.set_xlabel('Scale (Frequency)', fontsize=11)
        ax_gen.set_ylabel('Generalization Accuracy (%)', fontsize=11)
        ax_gen.set_title(f'n = {n}', fontsize=13)
        ax_gen.legend(loc='best', fontsize=9)
        ax_gen.grid(True, alpha=0.3)
        ax_gen.set_xscale('log')
        ax_gen.set_ylim([0, 105])
    
    axes[0, 0].set_ylabel('IDENTIFICATION\nAccuracy (%)', fontsize=12)
    axes[1, 0].set_ylabel('GENERALIZATION\nAccuracy (%)', fontsize=12)
    
    plt.suptitle('Effect of Grid Code Frequency on Task Performance\n(Low scale = coarse grid, High scale = fine grid)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_structure_preservation(structure_results, scales):
    """
    Visualize how structure preservation correlates with scale.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Bar plot of correlations
    ax1 = axes[0]
    correlations = [structure_results[s]['correlation'] for s in scales]
    bars = ax1.bar(range(len(scales)), correlations, color=plt.cm.viridis(np.linspace(0, 0.9, len(scales))))
    ax1.set_xticks(range(len(scales)))
    ax1.set_xticklabels([str(s) for s in scales])
    ax1.set_xlabel('Scale (Frequency)', fontsize=12)
    ax1.set_ylabel('Correlation (2D dist vs Encoded dist)', fontsize=12)
    ax1.set_title('Structure Preservation by Frequency\n(Higher = better preservation = less compression)', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, corr in zip(bars, correlations):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{corr:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Right: Scatter plot of 2D vs encoded distances for different scales
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(scales)))
    
    for idx, scale in enumerate(scales):
        # Subsample for clarity
        n_samples = min(500, len(structure_results[scale]['distances_2d']))
        indices = np.random.choice(len(structure_results[scale]['distances_2d']), n_samples, replace=False)
        
        ax2.scatter(
            structure_results[scale]['distances_2d'][indices],
            structure_results[scale]['distances_encoded'][indices],
            alpha=0.3, s=10, color=colors[idx],
            label=f'Scale={scale}'
        )
    
    ax2.set_xlabel('2D Euclidean Distance', fontsize=12)
    ax2.set_ylabel('Encoded Cosine Distance', fontsize=12)
    ax2.set_title('Distance Preservation Scatter\n(Tighter = better preservation)', fontsize=12)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig