# camera_calibration.py
"""
Stores camera intrinsic parameters obtained from ROS calibration for Tello drone.
"""
import numpy as np

# Calibrated values from ROS tools (replace placeholders)
fx = 921.170702
fy = 919.018377
cx = 459.904354
cy = 351.238301

# Camera Matrix (K)
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)

# Distortion Coefficients (k1, k2, p1, p2, k3)
# Format: [k1, k2, p1, p2, k3]
k1 = -0.033458
k2 = 0.105152
p1 = 0.001256
p2 = -0.006647
k3 = 0.000000 # Assuming k3 is the 5th value

# Ensure it's a row or column vector as OpenCV expects (e.g., (1,5) or (5,1))
# dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32) # Shape (5,)
dist_coeffs = np.array([[k1, k2, p1, p2, k3]], dtype=np.float32) # Shape (1, 5) is common

print("--- Using Calibrated Camera Intrinsics ---")
print("K:\n", camera_matrix)
print("Distortion Coefficients (k1, k2, p1, p2, k3):\n", dist_coeffs)
print("------------------------------------------")