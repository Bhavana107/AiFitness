import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the 2D joint angle (in degrees) at vertex point 'b' 
    formed by outer points 'a' and 'c'.
    
    Args:
        a (tuple or np.ndarray): Coordinates of first outer joint [x, y].
        b (tuple or np.ndarray): Coordinates of middle vertex joint [x, y].
        c (tuple or np.ndarray): Coordinates of second outer joint [x, y].
        
    Returns:
        float: Calculated angle in degrees, in the range [0.0, 180.0].
    """
    # Cast points to NumPy arrays
    np_a = np.array(a)
    np_b = np.array(b)
    np_c = np.array(c)
    
    # Form vectors relative to vertex 'b'
    ba = np_a - np_b
    bc = np_c - np_b
    
    # Calculate magnitudes
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    # Guard against division-by-zero (e.g. overlapping landmarks)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
        
    # Calculate cosine of the angle via dot product
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    
    # Clip cosine value to range [-1.0, 1.0] to avoid float precision issues with arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Retrieve the angle in radians and convert to degrees
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    
    return float(angle_deg)
