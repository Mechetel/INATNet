import numpy as np

# ==================== SRM Filter Initialization ====================

def get_srm_kernels():
    """
    Initialize SRM (Spatial Rich Model) high-pass filter kernels
    Using the exact filters from the provided configuration
    """
    # Filter Class 1 (8 filters) - First-order derivatives
    filter_class_1 = [
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
    ]
    
    # Filter Class 2 (4 filters) - Second-order derivatives (normalized by 2)
    filter_class_2 = [
        np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], dtype=np.float32) / 2,
        np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32) / 2,
        np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=np.float32) / 2,
        np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32) / 2,
    ]
    
    # Filter Class 3 (8 filters) - Third-order derivatives (normalized by 3)
    filter_class_3 = [
        np.array([[-1, 0, 0, 0, 0], [0, 3, 0, 0, 0], [0, 0, -3, 0, 0], 
                  [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 3,
        np.array([[0, 0, -1, 0, 0], [0, 0, 3, 0, 0], [0, 0, -3, 0, 0], 
                  [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 3,
        np.array([[0, 0, 0, 0, -1], [0, 0, 0, 3, 0], [0, 0, -3, 0, 0], 
                  [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 3,
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -3, 3, -1], 
                  [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 3,
        np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -3, 0, 0], 
                  [0, 0, 0, 3, 0], [0, 0, 0, 0, -1]], dtype=np.float32) / 3,
        np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -3, 0, 0], 
                  [0, 0, 3, 0, 0], [0, 0, -1, 0, 0]], dtype=np.float32) / 3,
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, -3, 0, 0], 
                  [0, 3, 0, 0, 0], [-1, 0, 0, 0, 0]], dtype=np.float32) / 3,
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 3, -3, 1, 0], 
                  [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 3
    ]
    
    # Edge filters 3x3 (4 filters) - normalized by 4
    filter_edge_3x3 = [
        np.array([[-1, 2, -1], [2, -4, 2], [0, 0, 0]], dtype=np.float32) / 4,
        np.array([[0, 2, -1], [0, -4, 2], [0, 2, -1]], dtype=np.float32) / 4,
        np.array([[0, 0, 0], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4,
        np.array([[-1, 2, 0], [2, -4, 0], [-1, 2, 0]], dtype=np.float32) / 4,
    ]
    
    # Square 3x3 (1 filter) - normalized by 4
    square_3x3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4
    
    # Edge filters 5x5 (4 filters) - normalized by 12
    filter_edge_5x5 = [
        np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], 
                  [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 12,
        np.array([[0, 0, -2, 2, -1], [0, 0, 8, -6, 2], [0, 0, -12, 8, -2], 
                  [0, 0, 8, -6, 2], [0, 0, -2, 2, -1]], dtype=np.float32) / 12,
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-2, 8, -12, 8, -2], 
                  [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32) / 12,
        np.array([[-1, 2, -2, 0, 0], [2, -6, 8, 0, 0], [-2, 8, -12, 0, 0], 
                  [2, -6, 8, 0, 0], [-1, 2, -2, 0, 0]], dtype=np.float32) / 12,
    ]
    
    # Square 5x5 (1 filter) - normalized by 12
    square_5x5 = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], 
                           [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32) / 12
    
    # Combine all filters (total: 8 + 4 + 8 + 4 + 1 + 4 + 1 = 30 filters)
    all_filters = filter_class_1 + filter_class_2 + filter_class_3 + \
                  filter_edge_3x3 + [square_3x3] + filter_edge_5x5 + [square_5x5]
    
    return all_filters