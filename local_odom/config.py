# config.py
"""Configuration constants for the Tello Advanced Controller."""

# Update Intervals (seconds)
UPDATE_INTERVAL_STATE = 0.1
UPDATE_INTERVAL_VIDEO = 0.03 # Aim for ~30fps
UPDATE_INTERVAL_PLOT = 0.1
UPDATE_INTERVAL_COMMAND_QUEUE = 1.0 # Timeout for checking command queue

# Control Parameters
CONTROL_DISTANCE_CM = 30
CONTROL_ANGLE_DEG = 30

# Plotting Parameters
PLOT_ORIENTATION_VECTOR_LENGTH = 20 # Length of the yaw indicator line in cm
PLOT_TRAJECTORY_LENGTH = 150 # Max number of points to keep for trajectory
PLOT_INITIAL_LIMITS = {
    'xlim': (-100, 100),
    'ylim': (-100, 100),
    'zlim': (0, 150)
}
PLOT_PADDING_FACTOR = 0.1 # Percentage padding for auto-scaling
PLOT_MIN_PADDING_XYZ = (50, 50, 20) # Minimum padding in cm

# Video Parameters
VIDEO_MAX_WIDTH = 640
VIDEO_MAX_HEIGHT = 480

# Tello Delays (seconds) - Added for stability
TELLO_STREAMON_DELAY = 0.5 # Delay after streamon() before reading frames
TELLO_COMMAND_DELAY = 0.5 # Delay after sending 'command'
TELLO_LAND_TIMEOUT = 5.0 # Generous time to wait for landing confirmation during disconnect

# Feature Detection Parameters
FEATURE_DETECTION_ENABLE_BY_DEFAULT = False
FEATURE_DETECTION_MAX_FEATURES = 300 # Number of features ORB tries to detect

# Visual Odometry Parameters
VO_ENABLE_BY_DEFAULT = False
VO_TRAJECTORY_LENGTH = 200 # Max points for VO trajectory plot
VO_PROCESS_FRAME_WIDTH = 640 # Resize frame before VO processing? (Optional, for performance)
VO_PROCESS_FRAME_HEIGHT = 480