import numpy as np
lx = 0.1
ly = 0.025 # ONLY ONE LAYER! so ly/2
def calculate_distortion_angle(u_bottom_right, u_top_right):

    C_original = np.array([lx, 0])          # Bottom-right (original)
    B_original = np.array([lx, ly])     # Top-right (original)
    
    C_distorted = C_original + u_bottom_right    # Bottom-right (distorted)
    B_distorted = B_original + u_top_right       # Top-right (distorted)
    
    # Original edge vector (vertical: from C to B)
    original_vector = B_original - C_original    # [0, ly]
    
    # Distorted edge vector (from C' to B')
    distorted_vector = B_distorted - C_distorted
    
    # Compute angle
    dot_product = np.dot(original_vector, distorted_vector)
    original_magnitude = np.linalg.norm(original_vector)
    distorted_magnitude = np.linalg.norm(distorted_vector)
    
    cos_theta = dot_product / (original_magnitude * distorted_magnitude)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.rad2deg(angle_rad)
    
    return angle_deg

u_bottom = np.array([-3.67951604e-05, 7.22699579e-06])
u_top = np.array([-4.15484239e-05, -2.91361137e-05])

angle = calculate_distortion_angle(u_bottom, u_top)
print(f"Distortion angle: {angle:.4f} degrees")