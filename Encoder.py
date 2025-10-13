import numpy as np

class GridEncoder:
    """
    Description: A GridEncoder models grid-cell-like spatial encoding by projecting 2D input coordinates 
                onto multiple periodic grids defined by modules, orientations, and phase shifts.
    """
    def __init__(self, n_modules=3, n_orientations=3, n_cells_per_orientation=5, scales=None):
        """
        Description: Initializes the GridEncoder with a specified number of modules, orientations 
                     and cells per orientation. Each module corresponds to a distinct spatial frequency,
                     and each orientation corresponds to a direction in the 2D plane.
                     Random phase shifts are used to differentiate grid cells within each module.

        ========= Parameters =========
        @param n_modules: Number of grid modules.
        @param n_orientations: Number of orientations per module.
        @param n_cells_per_orientation: Number of grid cells per orientation direction.
        @param scales: Custom list of scales for each module.

        ========= Returns =========
        @return None
        """
        self.n_modules = n_modules
        self.n_orientations = n_orientations
        self.n_cells_per_orientation = n_cells_per_orientation
        
        # Define orientations
        # In case of 3 orientations, the angles are 0, 60 and 120
        angles = np.linspace(0, np.pi, n_orientations, endpoint=False)
        self.orientations = np.stack([np.cos(angles), np.sin(angles)], axis=1) # A (n_orientations, 2) matrix. Each row is a unit orientation vector
        
        # Scales for each module
        self.scales = scales or np.sqrt(np.e) ** np.arange(n_modules)
        
        # Random phase shifts
        # P cells per orientation, O orientations per module, and M modules (M, O, P) matrix.
        self.phases = 2 * np.pi * np.random.rand(n_modules, n_orientations, n_cells_per_orientation) 
    
    def encode(self, x):
        """
        Description: Encodes a 2D input position into a high-dimensional representation
        based on periodic grid-cell-like responses.

        ========= Parameters =========
        @param x: 2D input position.

        ========= Returns =========
        @return encoded: Encoded grid-cell representation with shape (1, 2 * n_modules * n_orientations * n_cells_per_orientation).
        """
        outputs = []
        for m, f in enumerate(self.scales):
            for o, k in enumerate(self.orientations):
                proj = 2 * np.pi * f * (x @ k)
                for p in range(self.n_cells_per_orientation):
                    phi = self.phases[m, o, p]
                    outputs.append(np.cos(proj + phi))
                    outputs.append(np.sin(proj + phi))
        return np.concatenate(outputs, axis=1)
