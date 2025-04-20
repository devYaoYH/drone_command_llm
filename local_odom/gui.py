# gui.py
"""Defines the main Tello GUI Application class."""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import cv2
from PIL import Image, ImageTk
from djitellopy import Tello
import math
import queue
import numpy as np
import mediapipe as mp

import config # Import shared configuration
from plotter import PlotManager # Import the plot manager

from visual_odometry import VisualOdometry # Import the VO class
import camera_calibration

# MediaPipe Setup (early definition for clarity)
from mediapipe.tasks.python import vision as mp_vision
import os

# Define Model Path (Relative to script location)
MODEL_FILENAME = 'efficientdet_lite0.tflite'
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

# MediaPipe Drawing Utils (Keep as is, might still be useful)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class TelloGUIApp:
    """Main application class for Tello control GUI."""

    def __init__(self, root):
        self.root = root
        self.root.title("Tello Advanced Controller & Digital Twin")
        # self.root.geometry("1200x800") # Uncomment/adjust if needed

        # Tello State
        self.tello = None
        self._is_connected = False
        self._is_streaming = False
        self._is_flying = False
        self._stop_event = threading.Event()

        # Odometry & State Variables (Command-based)
        self.estimated_x = 0.0
        self.estimated_y = 0.0
        self.estimated_z = 0.0
        self.estimated_yaw = 0.0
        self.trajectory = []

        # Raw Sensor/State Data
        self.current_pitch = 0
        self.current_roll = 0
        self.current_yaw = 0
        self.current_vgx = 0
        self.current_vgy = 0
        self.current_vgz = 0
        self.current_tof = 0
        self.current_h = 0
        self.current_bat = 0
        self.current_time = 0

        # --- ADD INITIALIZATION FOR STATE STRINGVARS ---
        self.status_var = tk.StringVar(value="Status: Disconnected")
        self.battery_var = tk.StringVar(value="Battery: N/A")
        self.height_var = tk.StringVar(value="Height AGL: N/A cm")
        self.tof_var = tk.StringVar(value="TOF: N/A cm")
        self.attitude_var = tk.StringVar(value="Attitude (P,R,Y): N/A")
        self.speed_var = tk.StringVar(value="Speed (X,Y,Z): N/A")
        self.time_var = tk.StringVar(value="Flight Time: N/A")
        self.odom_var = tk.StringVar(value="Est. Odom (X,Y,Z,Yaw): 0.0, 0.0, 0.0, 0.0°")
        # --- END OF ADDED INITIALIZATIONS ---

        # Threading & Command Queue
        self.video_thread = None
        self.state_thread = None
        self.plot_thread = None
        self.command_queue = queue.Queue()
        self.command_thread = None
        self.frame_read = None

        # Visual Odometry
        self.vo_enabled = tk.BooleanVar(value=config.VO_ENABLE_BY_DEFAULT)
        self.vo_processor = None
        self.vo_trajectory = []
        self.vo_status_var = tk.StringVar(value="VO Status: Disabled") # This one was already present
        self._initialize_vo()

        # --- MediaPipe Object Detection ---
        self.object_detection_enabled = tk.BooleanVar(value=False)
        self.object_detector = None 
        self.object_detection_status_var = tk.StringVar(value="Obj Detect: Disabled")
        self._initialize_object_detector() 

        # Feature Detection
        self.orb_detector = None
        self.feature_detection_enabled = tk.BooleanVar(value=config.FEATURE_DETECTION_ENABLE_BY_DEFAULT)
        self._initialize_feature_detector()

        # GUI Setup
        self._setup_gui_layout()
        self._create_gui_widgets() # Now expects the StringVars to exist

        # Plotter Setup
        self.plot_manager = PlotManager(self.plot_frame)

        # Graceful Shutdown
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- GUI Setup Methods ---
    def _initialize_object_detector(self):
        """Initializes the MediaPipe Object Detection model using Tasks API."""
        print("Initializing MediaPipe Object Detection (Tasks API)...")
        if not os.path.exists(MODEL_PATH):
             error_msg = f"ERROR: Model file not found at {MODEL_PATH}"
             print(error_msg)
             self.object_detector = None
             self.object_detection_enabled.set(False)
             if hasattr(self, 'obj_detect_checkbox'):
                 self.obj_detect_checkbox.config(state=tk.DISABLED)
             self.object_detection_status_var.set(f"Obj Detect: Failed (No Model)")
             messagebox.showerror("MediaPipe Init Error", error_msg)
             return

        try:
            # 1. Create ObjectDetectorOptions
            base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
            options = mp_vision.ObjectDetectorOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE, # Process frame by frame
                score_threshold=0.5, # Confidence threshold
                # Optional: max_results=5, category_allowlist=['person']
            )

            # 2. Create the detector
            self.object_detector = mp_vision.ObjectDetector.create_from_options(options)
            print("MediaPipe Object Detection Initialized.")
            if hasattr(self, 'obj_detect_checkbox'):
                self.obj_detect_checkbox.config(state=tk.NORMAL)

        except Exception as e:
            print(f"ERROR: Failed to initialize MediaPipe Object Detection: {e}")
            import traceback
            traceback.print_exc() # Print full traceback
            self.object_detector = None
            self.object_detection_enabled.set(False)
            if hasattr(self, 'obj_detect_checkbox'):
                self.obj_detect_checkbox.config(state=tk.DISABLED)
            self.object_detection_status_var.set("Obj Detect: Init Failed")
            messagebox.showerror("MediaPipe Init Error", f"Failed to initialize MediaPipe:\n{e}")

    def _initialize_vo(self):
        """Initializes the Visual Odometry processor."""
        try:
            print("Initializing Visual Odometry...")
            self.vo_processor = VisualOdometry(
                cam_matrix=camera_calibration.camera_matrix,
                distortion=camera_calibration.dist_coeffs
            )
            print("Visual Odometry Initialized.")
            if hasattr(self, 'vo_checkbox'): # Ensure checkbox exists
                self.vo_checkbox.config(state=tk.NORMAL)
            if self.vo_enabled.get():
                self.vo_status_var.set("VO Status: Enabled (Idle)")
        except Exception as e:
            print(f"ERROR: Failed to initialize Visual Odometry: {e}")
            self.vo_processor = None
            self.vo_enabled.set(False)
            if hasattr(self, 'vo_checkbox'):
                self.vo_checkbox.config(state=tk.DISABLED)
            self.vo_status_var.set("VO Status: Init Failed")

    def _initialize_feature_detector(self):
        """Initializes the ORB feature detector."""
        try:
            # Create ORB detector instance
            # nfeatures: Max number of features to retain.
            # scaleFactor: Pyramid decimation ratio, > 1.
            # nlevels: Number of pyramid levels.
            # edgeThreshold: Size of the border where features are not detected.
            # WTA_K: Number of points used to produce BRIEF descriptor (2, 3, or 4).
            self.orb_detector = cv2.ORB_create(
                nfeatures=config.FEATURE_DETECTION_MAX_FEATURES,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=31,
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE, # Alternative: cv2.ORB_FAST_SCORE
                patchSize=31,
                fastThreshold=20
            )
            print(f"ORB detector initialized with max features: {config.FEATURE_DETECTION_MAX_FEATURES}")
        except AttributeError:
             print("ERROR: Failed to create ORB detector. Is OpenCV installed correctly (cv2.ORB_create exists)?")
             self.orb_detector = None # Ensure it's None if failed
             # Disable the feature checkbox if ORB failed
             self.feature_detection_enabled.set(False)
             if hasattr(self, 'feature_checkbox'): # Check if checkbox exists yet
                 self.feature_checkbox.config(state=tk.DISABLED)
        except Exception as e:
            print(f"ERROR: Unexpected error initializing ORB detector: {e}")
            self.orb_detector = None
            self.feature_detection_enabled.set(False)
            if hasattr(self, 'feature_checkbox'):
                 self.feature_checkbox.config(state=tk.DISABLED)

    def _setup_gui_layout(self):
        """Creates the main frames and configures grid layout."""
        self.root.columnconfigure(0, weight=1) # Left controls/state column
        self.root.columnconfigure(1, weight=3) # Right video/plot column (more space)
        self.root.rowconfigure(0, weight=1)    # Top row (controls/video)
        self.root.rowconfigure(1, weight=3)    # Bottom row (state/plot) - more space

        # Left Frame (Controls & State)
        left_frame = ttk.Frame(self.root, padding="10")
        left_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        left_frame.rowconfigure(0, weight=0) # Control frame - fixed size
        left_frame.rowconfigure(1, weight=1) # State frame - expandable
        left_frame.columnconfigure(0, weight=1)

        # Right Top Frame (Video)
        right_top_frame = ttk.Frame(self.root, padding="10")
        right_top_frame.grid(row=0, column=1, sticky="nsew")
        right_top_frame.columnconfigure(0, weight=1)
        right_top_frame.rowconfigure(0, weight=1)

        # Right Bottom Frame (Plot)
        right_bottom_frame = ttk.Frame(self.root, padding="10")
        right_bottom_frame.grid(row=1, column=1, sticky="nsew")
        right_bottom_frame.columnconfigure(0, weight=1)
        right_bottom_frame.rowconfigure(0, weight=1)

        # Assign frames to self for widget placement
        self.left_frame = left_frame
        self.right_top_frame = right_top_frame
        self.right_bottom_frame = right_bottom_frame

        # Create plot frame *here* so PlotManager can use it
        self.plot_frame = ttk.LabelFrame(self.right_bottom_frame, text="3D Digital Twin (Relative Odometry)", padding="5")
        self.plot_frame.grid(row=0, column=0, sticky="nsew")
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)


    def _create_gui_widgets(self):
        """Creates and places all the widgets in their respective frames."""

        # --- Control Frame Widgets (inside self.left_frame) ---
        control_frame = ttk.LabelFrame(self.left_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        # Configure columns for button expansion
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)

        # Row 1: Connect, Takeoff, Land
        self.connect_button = ttk.Button(control_frame, text="Connect", command=self.connect_tello)
        self.connect_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.takeoff_button = ttk.Button(control_frame, text="Take Off", command=self.takeoff, state=tk.DISABLED)
        self.takeoff_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.land_button = ttk.Button(control_frame, text="Land", command=self.land, state=tk.DISABLED)
        self.land_button.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        # Row 2: Movement Label
        ttk.Label(control_frame, text="Movement").grid(row=2, column=0, columnspan=3, pady=(10, 2))

        # Row 3, 4, 5: Directional Movement Buttons
        dist = config.CONTROL_DISTANCE_CM
        ttk.Button(control_frame, text="↑ Fwd", command=lambda: self._send_control_command('forward', dist)).grid(row=3, column=1, padx=5, pady=2, sticky="ew")
        ttk.Button(control_frame, text="← Left", command=lambda: self._send_control_command('left', dist)).grid(row=4, column=0, padx=5, pady=2, sticky="ew")
        ttk.Button(control_frame, text="→ Right", command=lambda: self._send_control_command('right', dist)).grid(row=4, column=2, padx=5, pady=2, sticky="ew")
        ttk.Button(control_frame, text="↓ Back", command=lambda: self._send_control_command('back', dist)).grid(row=5, column=1, padx=5, pady=2, sticky="ew")

        # Row 6: Vertical Movement Buttons
        ttk.Button(control_frame, text="⤒ Up", command=lambda: self._send_control_command('up', dist)).grid(row=6, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(control_frame, text="⤓ Down", command=lambda: self._send_control_command('down', dist)).grid(row=6, column=2, padx=5, pady=5, sticky="ew")

        # Row 7: Rotation Buttons
        angle = config.CONTROL_ANGLE_DEG
        ttk.Button(control_frame, text="↶ CCW", command=lambda: self._send_control_command('ccw', angle)).grid(row=7, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(control_frame, text="↷ CW", command=lambda: self._send_control_command('cw', angle)).grid(row=7, column=2, padx=5, pady=5, sticky="ew")

        # Row 8: Emergency Button
        self.emergency_button = ttk.Button(control_frame, text="! EMERGENCY !", command=self.emergency_land)
        self.emergency_button.grid(row=8, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

        # Row 9: Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=9, column=0, columnspan=3, sticky='ew', pady=5)

        # Row 10: Feature Detection Checkbox
        self.feature_checkbox = tk.Checkbutton(control_frame, text="Enable Feature Detection (ORB)",
                                               variable=self.feature_detection_enabled,
                                               command=self._on_feature_toggle)
        self.feature_checkbox.grid(row=10, column=0, columnspan=3, sticky='w', padx=5, pady=5)
        # Check if detector was initialized successfully (assuming it's None if failed)
        if not hasattr(self, 'orb_detector') or not self.orb_detector:
             self.feature_checkbox.config(state=tk.DISABLED)

        # Row 11: Visual Odometry Checkbox
        self.vo_checkbox = tk.Checkbutton(control_frame, text="Enable Visual Odometry (VO)",
                                          variable=self.vo_enabled,
                                          command=self._on_vo_toggle)
        self.vo_checkbox.grid(row=11, column=0, columnspan=3, sticky='w', padx=5, pady=5)
        # Check if VO processor was initialized successfully
        if not hasattr(self, 'vo_processor') or not self.vo_processor:
            self.vo_checkbox.config(state=tk.DISABLED)

        # Row 12: Object Detection Checkbox
        self.obj_detect_checkbox = tk.Checkbutton(control_frame, text="Enable Object Detection (MP)",
                                                 variable=self.object_detection_enabled,
                                                 command=self._on_object_detection_toggle)
        self.obj_detect_checkbox.grid(row=12, column=0, columnspan=3, sticky='w', padx=5, pady=5)
        # Check if object detector was initialized successfully
        if not hasattr(self, 'object_detector') or not self.object_detector:
            self.obj_detect_checkbox.config(state=tk.DISABLED)


        # --- State Frame Widgets (inside self.left_frame) ---
        # Create the state frame itself
        state_frame = ttk.LabelFrame(self.left_frame, text="State & Odometry", padding="10")
        state_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5) # Place it in the second row of left_frame
        state_frame.columnconfigure(1, weight=1) # Allow value column to expand

        # Assumes these StringVars are initialized in __init__
        # self.status_var = tk.StringVar(value="Status: Disconnected")
        # self.battery_var = tk.StringVar(value="Battery: N/A")
        # ... and so on for height_var, tof_var, attitude_var, speed_var, time_var, odom_var

        # Add state labels and their corresponding StringVars
        row_idx = 0
        ttk.Label(state_frame, text="Status:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(state_frame, textvariable=self.status_var).grid(row=row_idx, column=1, sticky="w", padx=5, pady=1); row_idx += 1

        ttk.Label(state_frame, text="Battery:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(state_frame, textvariable=self.battery_var).grid(row=row_idx, column=1, sticky="w", padx=5, pady=1); row_idx += 1

        ttk.Label(state_frame, text="Height AGL:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(state_frame, textvariable=self.height_var).grid(row=row_idx, column=1, sticky="w", padx=5, pady=1); row_idx += 1

        ttk.Label(state_frame, text="TOF:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(state_frame, textvariable=self.tof_var).grid(row=row_idx, column=1, sticky="w", padx=5, pady=1); row_idx += 1

        ttk.Label(state_frame, text="Attitude:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(state_frame, textvariable=self.attitude_var).grid(row=row_idx, column=1, sticky="w", padx=5, pady=1); row_idx += 1

        ttk.Label(state_frame, text="Speed:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(state_frame, textvariable=self.speed_var).grid(row=row_idx, column=1, sticky="w", padx=5, pady=1); row_idx += 1

        ttk.Label(state_frame, text="Flight Time:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(state_frame, textvariable=self.time_var).grid(row=row_idx, column=1, sticky="w", padx=5, pady=1); row_idx += 1

        # Command-based Odometry Display
        ttk.Label(state_frame, text="Cmd Odom:", font=("Arial", 9, "bold")).grid(row=row_idx, column=0, sticky="w", padx=5, pady=(5,1))
        ttk.Label(state_frame, textvariable=self.odom_var, font=("Arial", 9)).grid(row=row_idx, column=1, sticky="w", padx=5, pady=(5,1)); row_idx += 1

        # Odometry Notes (Optional, could be removed for space)
        # ttk.Label(state_frame, text="Note: Odometry is command-based estimate.", font=("Arial", 8, "italic")).grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=1); row_idx += 1
        # ttk.Label(state_frame, text="      Subject to drift. Z primarily uses TOF.", font=("Arial", 8, "italic")).grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=1); row_idx += 1

        # Add the VO Status Label
        # Assumes self.vo_status_var = tk.StringVar(...) in __init__
        ttk.Label(state_frame, textvariable=self.vo_status_var, font=("Arial", 9, "italic")).grid(
            row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=(5,1))
        row_idx += 1 # Increment row index

        # Add the Object Detection Status Label
        # Assumes self.object_detection_status_var = tk.StringVar(...) in __init__
        ttk.Label(state_frame, textvariable=self.object_detection_status_var, font=("Arial", 9, "italic")).grid(
            row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=(5,1))
        row_idx += 1 # Increment row index

        # --- Video Frame Widgets (inside self.right_top_frame) ---
        video_frame = ttk.LabelFrame(self.right_top_frame, text="Video Stream", padding="5")
        video_frame.grid(row=0, column=0, sticky="nsew")
        # Configure row/column weights for video label expansion
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(video_frame, text="Video Feed Disconnected", anchor=tk.CENTER)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # --- 3D Plot Frame ---
        # Reminder: The actual plot canvas (managed by PlotManager) is added
        # to self.plot_frame (created in _setup_gui_layout) during the
        # TelloGUIApp initialization (__init__). No widgets needed here for the plot itself.

    def _on_feature_toggle(self):
        """Callback function when the feature detection checkbox is toggled."""
        if self.feature_detection_enabled.get():
            if not self.orb_detector:
                messagebox.showwarning("Feature Detection", "ORB detector was not initialized successfully. Cannot enable.")
                self.feature_detection_enabled.set(False) # Force disable
            else:
                print("ORB Feature Detection Enabled.")
        else:
            print("ORB Feature Detection Disabled.")

    def _on_vo_toggle(self):
        """Callback for VO checkbox toggle."""
        if self.vo_enabled.get():
            if not self.vo_processor:
                messagebox.showwarning("Visual Odometry", "VO processor failed to initialize.")
                self.vo_enabled.set(False)
                self.vo_status_var.set("VO Status: Init Failed")
            else:
                print("Visual Odometry Enabled.")
                # Reset VO state when enabling? Optional.
                # self.vo_processor.reset()
                # self.vo_trajectory = self.vo_processor.get_trajectory()
                self.vo_status_var.set("VO Status: Enabled (Running)")
        else:
            print("Visual Odometry Disabled.")
            self.vo_status_var.set("VO Status: Disabled")
            # Clear VO trajectory from plot? Handled by plot_manager update.
    
    def _on_object_detection_toggle(self):
        """Callback for Object Detection checkbox toggle."""
        if self.object_detection_enabled.get():
            if not self.object_detector:
                messagebox.showwarning("Object Detection", "MediaPipe detector failed to initialize.")
                self.object_detection_enabled.set(False)
                self.object_detection_status_var.set("Obj Detect: Init Failed")
            else:
                print("MediaPipe Object Detection Enabled.")
                self.object_detection_status_var.set("Obj Detect: Enabled")
        else:
            print("MediaPipe Object Detection Disabled.")
            self.object_detection_status_var.set("Obj Detect: Disabled")


    # --- Tello Connection and Control ---

    def connect_tello(self):
        if self._is_connected: return
        try:
            self._update_status("Connecting...")
            self.root.update_idletasks()

            self.tello = Tello()
            self.tello.connect()
            self._is_connected = True
            print("Tello Connected.")

            if self.vo_processor:
                self.vo_processor.reset()
                self.vo_trajectory = self.vo_processor.get_trajectory()

            # Reset state before starting threads/commands
            self._reset_odometry_and_plot()

            self._update_status(f"Connected (Battery: {self.tello.get_battery()}%)")
            self.connect_button.config(text="Disconnect", command=self.disconnect_tello)
            self.takeoff_button.config(state=tk.NORMAL)
            self.land_button.config(state=tk.NORMAL) # Land is always possible if connected

            # Start video stream
            print("Starting video stream...")
            self.tello.streamon()
            time.sleep(config.TELLO_STREAMON_DELAY) # **Critical Delay**
            self.frame_read = self.tello.get_frame_read()
            # **Error Check**: Check if frame_read initialized correctly
            if self.frame_read is None:
                 print("ERROR: Failed to get frame reader. Video stream may not work.")
                 messagebox.showerror("Video Error", "Failed to initialize video stream reader.")
                 # Decide if connection should proceed without video
                 # self.disconnect_tello() # Option: Disconnect if video fails
                 # return
            else:
                self._is_streaming = True
                print("Video stream started.")


            # Start background threads
            self._stop_event.clear()
            self._start_threads()

        except Exception as e:
            print(f"Connection failed: {e}")
            messagebox.showerror("Connection Error", f"Failed to connect to Tello.\nEnsure Tello is on and you are connected to its Wi-Fi.\nError: {e}")
            self._update_status("Connection Failed")
            self.tello = None
            self._is_connected = False

    def disconnect_tello(self):
        if not self._is_connected: return
        print("Disconnecting...")
        should_land = self._is_flying # Check if landing is needed *before* sending commands

        try:
            # Stop drone movement and land if necessary
            if should_land:
                print("Landing before disconnect...")
                # Send land command directly (might be safer than queue if closing)
                try:
                    self.tello.land()
                    # Give it time to land, but don't wait indefinitely if unresponsive
                    land_wait_start = time.time()
                    while self._is_flying and (time.time() - land_wait_start) < config.TELLO_LAND_TIMEOUT:
                        # Check actual flight state if possible (might be unreliable during disconnect)
                         try:
                             if self.tello.get_height() <= 15 and self.tello.get_speed_z() == 0 : # Crude check
                                  self._is_flying = False
                                  print("Landing detected.")
                                  break
                         except Exception: pass # Ignore errors during this check
                         time.sleep(0.2)
                    if self._is_flying:
                         print("Landing timeout reached or check failed, forcing emergency.")
                         self.tello.emergency() # Force stop if landing unclear
                    self._is_flying = False
                
                except Exception as land_err:
                    print(f"Error sending land/emergency during disconnect: {land_err}. Attempting emergency again.")
                    try: self.tello.emergency()
                    except Exception as emer_err: print(f"Emergency command failed: {emer_err}")
                self._is_flying = False # Assume stopped after commands

            # Stop streaming
            if self._is_streaming:
                try:
                    print("Stopping video stream...")
                    self.tello.streamoff()
                except Exception as e: print(f"Stream off error: {e}")
                self._is_streaming = False

            # Signal threads to stop
            self._stop_event.set()
            # Signal command thread specifically
            if self.command_thread and self.command_thread.is_alive():
                 self.command_queue.put(None) # Sentinel value

            # Wait for threads to finish
            print("Joining background threads...")
            threads = [self.video_thread, self.state_thread, self.plot_thread, self.command_thread]
            for t in threads:
                if t and t.is_alive():
                    t.join(timeout=1.5) # Wait max 1.5s per thread
            print("Threads joined.")

            # Reset VO state on disconnect
            if self.vo_processor:
                self.vo_processor.reset()
                self.vo_trajectory = self.vo_processor.get_trajectory()
                if self.vo_enabled.get(): # Update status if it was enabled
                    self.vo_status_var.set("VO Status: Enabled (Idle)")

            # Tello cleanup (optional, socket should close)
            # try: self.tello.end()
            # except Exception as e: print(f"Tello end error: {e}")

        except Exception as e:
            print(f"Error during disconnect procedure: {e}")
            try:
                # Final fallback if anything failed
                if self.tello: self.tello.emergency()
            except Exception: pass
        finally:
            # Reset all states regardless of errors
            self.tello = None
            self.frame_read = None
            self._is_connected = False
            self._is_flying = False
            self._is_streaming = False
            self._stop_event.clear() # Clear event for next connection

            # Reset GUI
            self._update_status("Disconnected")
            self._reset_gui_state()
            self.plot_manager.reset_plot()
            print("Tello Disconnected.")

    def takeoff(self):
        if self._is_connected and not self._is_flying:
            print("Attempting Takeoff...")
            # Reset position just before takeoff
            initial_tof = 0
            try:
                # Use 'command' first to ensure SDK mode is active
                self._send_command_via_queue("command")
                time.sleep(config.TELLO_COMMAND_DELAY) # Wait for SDK activation
                initial_tof = self.tello.get_distance_tof()
                print(f"Initial TOF before takeoff: {initial_tof} cm")
            except Exception as e:
                print(f"Warning: Failed to get initial TOF or send 'command': {e}")
            self._reset_odometry_and_plot(initial_z=max(0, initial_tof)) # Use TOF if valid, else 0

            if self.vo_processor:
                self.vo_processor.reset()
                self.vo_trajectory = self.vo_processor.get_trajectory()
                # If VO is enabled, update status
                if self.vo_enabled.get():
                    self.vo_status_var.set("VO Status: Enabled (Running)")

            # Send takeoff command via queue
            if self._send_command_via_queue("takeoff"):
                 self._is_flying = True # Assume takeoff command sent successfully
                 print("Takeoff command sent.")
                 self._update_status("Flying") # Update status immediately
            else:
                 print("Failed to queue takeoff command.")
                 messagebox.showerror("Takeoff Error", "Could not send takeoff command.")


    def land(self):
        if self._is_connected and self._is_flying:
            print("Attempting Land...")
            if self._send_command_via_queue("land"):
                 # Assume landing starts, actual state change handled by state updates
                 # self._is_flying = False # Set this based on state update? Or assume for now? Let's assume.
                 self._is_flying = False # Assume landing will succeed
                 print("Land command sent.")
                 self._update_status("Landing...")
            else:
                print("Failed to queue land command.")
                messagebox.showerror("Land Error", "Could not send land command.")
        elif self._is_connected and not self._is_flying:
             print("Already landed.")


    def emergency_land(self):
         if self._is_connected:
            print("!!! EMERGENCY LANDING !!!")
            # Send emergency immediately, potentially bypassing queue for critical safety
            # Although Tello handles 'emergency' quickly, direct send might be marginally faster
            try:
                self.tello.emergency()
                self._is_flying = False # Assume it stops immediately
                self._update_status("Emergency Landed")
                print("Emergency command sent directly.")
            except Exception as e:
                print(f"Failed to send emergency command directly: {e}")
                # Fallback: try queueing it? Less ideal for emergency.
                # self._send_command_via_queue("emergency") # No, direct is better if possible
                messagebox.showerror("Emergency Error", f"Failed to send EMERGENCY command: {e}")
         else:
            print("Cannot send emergency: Not connected.")

    # --- Command Sending and Odometry ---

    def _send_command_via_queue(self, command_str):
        """Safely adds a command to the processing queue."""
        if not self._is_connected or not self.command_queue:
            print(f"Cannot queue command '{command_str}': Not connected or queue unavailable.")
            return False
        try:
            self.command_queue.put(command_str)
            # print(f"Queued command: {command_str}") # Optional: Verbose logging
            return True
        except Exception as e:
            print(f"Error queueing command '{command_str}': {e}")
            return False

    def _send_control_command(self, direction, value):
        """Sends a movement/rotation command and updates estimated odometry."""
        if not self._is_connected:
            print("Cannot send control command: Not connected.")
            return
        if not self._is_flying and direction not in ['cw', 'ccw']: # Allow rotation on ground? Maybe not.
             print("Cannot send movement command: Not flying.")
             # Allow rotation commands even if not flying? Tello might ignore them.
             # Let's restrict all control commands to flying state for simplicity.
             if direction in ['cw', 'ccw']:
                 print("Cannot send rotation command: Not flying.")
             return


        command_str = f"{direction} {value}"
        if self._send_command_via_queue(command_str):
            # --- Update Estimated Odometry (Based on Command Sent) ---
            # Note: This is a simplified model and prone to drift.
            # It assumes commands execute perfectly. Z is updated by sensor.
            current_yaw_rad = math.radians(self.estimated_yaw)
            dx, dy, dz, dyaw = 0.0, 0.0, 0.0, 0.0

            # Using standard right-hand rule coordinate system:
            # +X: Forward, +Y: Left, +Z: Up
            # Yaw=0: Facing +X, Yaw=90: Facing +Y (Left)
            if direction == 'forward':
                dx = value * math.cos(current_yaw_rad)
                dy = value * math.sin(current_yaw_rad) # Forward moves along heading vector
            elif direction == 'back':
                dx = -value * math.cos(current_yaw_rad)
                dy = -value * math.sin(current_yaw_rad) # Back moves opposite to heading
            elif direction == 'left':
                # Left is 90deg CCW from forward: (-sin, cos) vector component
                dx = -value * math.sin(current_yaw_rad)
                dy = value * math.cos(current_yaw_rad)
            elif direction == 'right':
                 # Right is 90deg CW from forward: (sin, -cos) vector component
                dx = value * math.sin(current_yaw_rad)
                dy = -value * math.cos(current_yaw_rad)
            elif direction == 'up':
                pass # Z primarily updated by TOF sensor in state loop
            elif direction == 'down':
                pass # Z primarily updated by TOF sensor
            elif direction == 'cw': # Clockwise rotation decreases yaw (standard convention)
                dyaw = -value
            elif direction == 'ccw': # Counter-clockwise increases yaw
                dyaw = value

            # Update estimates
            self.estimated_x += dx
            self.estimated_y += dy
            # self.estimated_z += dz # Z handled by sensor update

            # Ensure yaw stays within [0, 360) or (-180, 180] - let's use [0, 360)
            self.estimated_yaw = (self.estimated_yaw + dyaw) % 360

            # Append to trajectory (using current sensor Z)
            self._add_trajectory_point()
            # Update GUI labels immediately after command sent (more responsive feel)
            self._update_odometry_display()

    def _add_trajectory_point(self):
        """Adds the current estimated point (using sensor Z) to the trajectory list."""
        # Use the most recent sensor-based Z for trajectory plotting
        current_point = (self.estimated_x, self.estimated_y, self.estimated_z)
        self.trajectory.append(current_point)
        # Limit trajectory length
        if len(self.trajectory) > config.PLOT_TRAJECTORY_LENGTH:
            self.trajectory.pop(0) # Remove the oldest point

    def _reset_odometry_and_plot(self, initial_z=0.0):
        """Resets estimated position, yaw, trajectory, and plot."""
        print(f"Resetting odometry. Initial Z set to: {initial_z}")
        self.estimated_x = 0.0
        self.estimated_y = 0.0
        self.estimated_z = initial_z # Use provided Z (e.g., from TOF)
        self.estimated_yaw = 0.0
        self.trajectory = [(0.0, 0.0, initial_z)] # Start trajectory at initial point
        self._update_odometry_display()
        if hasattr(self, 'plot_manager'):
            self.plot_manager.reset_plot()


    # --- Background Thread Management ---

    def _start_threads(self):
        """Starts all background threads."""
        if not self.state_thread or not self.state_thread.is_alive():
            self.state_thread = threading.Thread(target=self._state_update_loop, daemon=True)
            self.state_thread.start()

        if self._is_streaming and (not self.video_thread or not self.video_thread.is_alive()):
             # Only start video thread if streaming was successfully initiated
            if self.frame_read:
                self.video_thread = threading.Thread(target=self._video_update_loop, daemon=True)
                self.video_thread.start()
            else:
                print("Skipping video thread start: frame_read is None.")


        if not self.plot_thread or not self.plot_thread.is_alive():
            self.plot_thread = threading.Thread(target=self._plot_update_loop, daemon=True)
            self.plot_thread.start()

        if not self.command_thread or not self.command_thread.is_alive():
            self.command_thread = threading.Thread(target=self._command_processing_loop, daemon=True)
            self.command_thread.start()

    # --- Background Thread Loops ---

    def _video_update_loop(self):
        """Continuously reads frames, performs feature detection (if enabled), and schedules GUI updates."""
        print("Video update thread started.")
        frame_count = 0
        last_print_time = time.time()

        if not self.frame_read:
            print("Video thread exiting: frame_read object not available.")
            self.root.after(0, self._update_video_label_text, "Video Init Failed")
            return

        while not self._stop_event.is_set():
            # try:
            #     if not self._is_connected or not self._is_streaming:
            #         print("Video thread stopping: Disconnected or not streaming.")
            #         break
            #     if self.frame_read.stopped:
            #         print("Video thread stopping: Tello video source stopped.")
            #         break

            frame = self.frame_read.frame
            if frame is None:
                time.sleep(0.01)
                continue

                # Make a mutable copy of the frame to draw on
            display_frame = frame.copy()

            try:
                # 1. --- Visual Odometry Processing ---
                vo_processed = False
                current_tracking_quality = 0.0
                if self.vo_enabled.get() and self.vo_processor:
                    # ... (resize vo_input_frame if needed) ...
                    vo_input_frame = display_frame # Use the current frame copy

                    current_altitude_m = self.current_tof / 100.0 if self.current_tof is not None and self.current_tof > 0 else None
                    vo_R, vo_t, vo_output_frame, current_tracking_quality = self.vo_processor.process_frame(
                        vo_input_frame, altitude=current_altitude_m
                    )
                    # Use the frame returned by VO (which might have visualizations)
                    display_frame = vo_output_frame.copy() # Use the frame returned by VO
                    vo_processed = True

                    if vo_R is not None: # Update trajectory if VO succeeded
                        self.vo_trajectory = self.vo_processor.get_trajectory()
                        if len(self.vo_trajectory) > config.VO_TRAJECTORY_LENGTH:
                            self.vo_trajectory = self.vo_trajectory[-config.VO_TRAJECTORY_LENGTH:]
                    self.root.after(0, self.vo_status_var.set, f"VO Status: Running (Qual: {current_tracking_quality:.2f})")


                 # 2. --- MediaPipe Object Detection (Updated Processing) ---
                detected_objects_info = []
                mp_detected = False # Flag if MediaPipe ran
                if self.object_detection_enabled.get() and self.object_detector:
                    mp_detected = True
                    try:
                        # Convert BGR frame to MediaPipe Image format
                        # NOTE: Needs RGB format for MediaPipe!
                        image_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                        # --- Perform detection ---
                        detection_result = self.object_detector.detect(mp_image)

                        # --- Process and Draw Results ---
                        if detection_result.detections:
                            self.root.after(0, self.object_detection_status_var.set, f"Obj Detect: {len(detection_result.detections)} found")
                            # Use a utility function to draw (or draw manually)
                            display_frame = self._draw_mp_detections(display_frame, detection_result)

                            # --- Extract Information (Updated Access) ---
                            for detection in detection_result.detections:
                                if not detection.categories: continue # Skip if no category info
                                category = detection.categories[0] # Usually best score is first
                                label = category.category_name
                                score = category.score
                                bbox = detection.bounding_box # Access bounding_box directly

                                xmin = bbox.origin_x
                                ymin = bbox.origin_y
                                box_w = bbox.width
                                box_h = bbox.height
                                center_x = xmin + box_w // 2
                                center_y = ymin + box_h // 2

                                detected_objects_info.append({
                                    'label': label,
                                    'score': score,
                                    'box_abs': (xmin, ymin, box_w, box_h),
                                    'center_abs': (center_x, center_y)
                                })
                        else:
                             self.root.after(0, self.object_detection_status_var.set, "Obj Detect: None found")

                    except Exception as e_mp:
                         print(f"Error during MediaPipe Object Detection: {e_mp}")
                         self.root.after(0, self.object_detection_status_var.set, "Obj Detect: Error")


                # 3. --- Feature Detection Drawing (Optional Redundancy Check) ---
                # Consider disabling this if VO and/or Object Detection are enabled,
                # as the frame might get too cluttered and performance degrades.
                if self.feature_detection_enabled.get() and not vo_processed and self.orb_detector:
                     # Only draw if VO didn't already process/draw on the frame
                     gray_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                     keypoints = self.orb_detector.detect(gray_display, None)
                     display_frame = cv2.drawKeypoints(
                         display_frame, keypoints, None, color=(0, 255, 0), flags=0
                     )
                     cv2.putText(display_frame, f"ORB: {len(keypoints)}", (10, display_frame.shape[0] - 10), # Position near bottom
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


                # --- Frame Resizing and Display ---
                h, w, _ = display_frame.shape
                ratio = min(config.VIDEO_MAX_WIDTH / w, config.VIDEO_MAX_HEIGHT / h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                frame_resized = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                self.root.after(0, self._update_video_label_image, imgtk)

                # Convert color space for Tkinter (BGR to RGB)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                # Schedule GUI update on main thread
                self.root.after(0, self._update_video_label_image, imgtk)
                frame_count += 1

                # Optional FPS calculation
                current_time = time.time()
                if current_time - last_print_time >= 5.0:
                    fps = frame_count / (current_time - last_print_time)
                    status_str = f"Video approx FPS: {fps:.1f}"
                    if self.feature_detection_enabled.get():
                        status_str += " (Features ON)"
                    print(status_str)
                    frame_count = 0
                    last_print_time = current_time

                time.sleep(config.UPDATE_INTERVAL_VIDEO)

            except Exception as e:
                print(f"Error in video thread: {e}")
                self.root.after(0, self._update_video_label_text, f"Video Error: {e}")
                time.sleep(1)

        print("Video update thread finished.")
        self.root.after(0, self._update_video_label_text, "Video Feed Stopped")

    def _draw_mp_detections(self, image, detection_result):
        """Helper function to draw MediaPipe detection results onto the image."""
        # Colors and font settings
        MARGIN = 10  # pixels
        ROW_SIZE = 30  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        TEXT_COLOR = (0, 255, 0) # Green
        BOX_COLOR = (0, 0, 255) # Red
        BOX_THICKNESS = 1

        annotated_image = image.copy() # Work on a copy

        for detection in detection_result.detections:
            if not detection.categories: continue

            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, BOX_COLOR, BOX_THICKNESS)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            score = round(category.score, 2)
            result_text = f"{category_name} ({score})"
            text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image
    
    def _move_based_on_detection(self, detected_objects):
        """
        Placeholder function for implementing movement logic based on detections.
        THIS REQUIRES SIGNIFICANT CONTROL LOGIC DEVELOPMENT.
        """
        if not self._is_flying or not detected_objects:
             return # Only act if flying and objects are detected

        # Example: Find the first detected 'person'
        target_label = 'person'
        target_object = None
        for obj in detected_objects:
            if obj['label'] == target_label and obj['score'] > 0.6: # Confidence threshold
                 target_object = obj
                 break

        if target_object:
             # Basic Proportional Control Example:
             frame_center_x = config.VIDEO_MAX_WIDTH / 2 # Assuming VIDEO_MAX_WIDTH used for display
             frame_center_y = config.VIDEO_MAX_HEIGHT / 2

             obj_center_x = target_object['center_abs'][0] # Need center relative to processed frame size
             obj_center_y = target_object['center_abs'][1]

             # --- Control Logic Needs Careful Design ---
             # 1. Yaw Control: Center the object horizontally
             error_x = frame_center_x - obj_center_x
             yaw_speed = int(np.clip(error_x * 0.3, -50, 50)) # Scale error to yaw speed (Needs tuning!)

             # 2. Forward/Backward Control (based on size? or vertical pos?)
             # Example: Move forward if object is small (far), back if large (close)
             box_area = target_object['box_abs'][2] * target_object['box_abs'][3]
             target_area = (config.VIDEO_MAX_WIDTH * config.VIDEO_MAX_HEIGHT) * 0.1 # Target 10% of area (Needs tuning!)
             error_area = target_area - box_area
             fwd_speed = int(np.clip(error_area * 0.001, -30, 30)) # Scale area error to fwd speed (Needs tuning!)

             # 3. Up/Down Control: Center the object vertically
             error_y = frame_center_y - obj_center_y
             ud_speed = int(np.clip(error_y * -0.3, -40, 40)) # Note: OpenCV Y is down, Tello Up is positive (invert error)

             # Send RC command (Non-blocking) - Requires Tello EDU or specific firmware? Check djitellopy docs.
             # self.tello.send_rc_control(left_right=0, forward_backward=fwd_speed, up_down=ud_speed, yaw=yaw_speed)
             # OR Queue discrete commands (less smooth):
             print(f"Target '{target_label}': CenterX={obj_center_x}, ErrorX={error_x} => YawCmd={yaw_speed}")
             # Example: If error is large, send a turn command
             # if abs(error_x) > 50: # Pixel threshold
             #     if error_x > 0: # Object is to the left, turn left (CCW)
             #         self._send_command_via_queue(f"ccw {abs(yaw_speed)//5}") # Scale speed to angle
             #     else: # Object is to the right, turn right (CW)
             #          self._send_command_via_queue(f"cw {abs(yaw_speed)//5}")

             # Add similar logic for forward/back and up/down based on errors
             # VERY IMPORTANT: Add deadbands, smoothing, safety checks.

        # else:
             # Optional: If target lost, stop RC control
             # self.tello.send_rc_control(0, 0, 0, 0)
             pass # No target found or lost

    def _state_update_loop(self):
        """Periodically fetches Tello state and updates GUI."""
        print("State update thread started.")
        while not self._stop_event.is_set():
            try:
                if self._is_connected and self.tello:
                    # Fetch states individually with error handling
                    bat = self._get_tello_state(self.tello.get_battery, "battery")
                    h = self._get_tello_state(self.tello.get_height, "height")
                    tof = self._get_tello_state(self.tello.get_distance_tof, "TOF")
                    pitch = self._get_tello_state(self.tello.get_pitch, "pitch")
                    roll = self._get_tello_state(self.tello.get_roll, "roll")
                    yaw = self._get_tello_state(self.tello.get_yaw, "yaw")
                    vgx = self._get_tello_state(self.tello.get_speed_x, "speed_x")
                    vgy = self._get_tello_state(self.tello.get_speed_y, "speed_y")
                    vgz = self._get_tello_state(self.tello.get_speed_z, "speed_z")
                    flight_time = self._get_tello_state(self.tello.get_flight_time, "flight_time")

                    # Update internal state variables
                    self.current_bat = bat if bat is not None else self.current_bat
                    self.current_h = h if h is not None else self.current_h
                    self.current_tof = tof if tof is not None else self.current_tof
                    self.current_pitch = pitch if pitch is not None else self.current_pitch
                    self.current_roll = roll if roll is not None else self.current_roll
                    self.current_yaw = yaw if yaw is not None else self.current_yaw
                    self.current_vgx = vgx if vgx is not None else self.current_vgx
                    self.current_vgy = vgy if vgy is not None else self.current_vgy
                    self.current_vgz = vgz if vgz is not None else self.current_vgz
                    self.current_time = flight_time if flight_time is not None else self.current_time

                    # --- Update estimated Z based on TOF sensor ---
                    # Only update if flying and TOF reading is plausible
                    # Tello TOF range is roughly 30cm to 1000cm (or more), but errors occur at edges.
                    # Use a reasonable range like 10cm to 500cm? Adjust as needed.
                    if self._is_flying and tof is not None and 10 < tof < 800: # Basic validity check
                        # Low-pass filter might be good here to smooth Z estimates
                        # Simple moving average or exponential smoothing could be applied
                        # Example: Basic smoothing (adjust alpha)
                        alpha = 0.7 # Weight of new reading
                        self.estimated_z = alpha * tof + (1 - alpha) * self.estimated_z
                        # self.estimated_z = tof # Direct update

                        # Update trajectory point if Z changed significantly?
                        # Or rely on plot update loop to show latest Z?
                        # Let's rely on plot update loop.

                    # Determine flight status based on state? (More reliable than just commands)
                    # Example: Consider landed if height < 15cm and Z speed near 0?
                    if self._is_flying and h is not None and h < 15 and vgz == 0:
                         print("State update suggests drone has landed.")
                         # self._is_flying = False # careful about race conditions with commands
                         # self._update_status("Landed (detected)") # maybe add a specific status

                    # Schedule GUI update
                    self.root.after(0, self._update_state_labels)
                else:
                    break # Stop loop if disconnected

                time.sleep(config.UPDATE_INTERVAL_STATE)
            except Exception as e:
                print(f"Error in state update thread: {e}")
                # Check if error is due to disconnection
                if not self._is_connected: break
                time.sleep(1) # Wait longer after error
        print("State update thread finished.")

    def _get_tello_state(self, func, name):
        """Helper to get a single state value with error handling."""
        try:
            return func()
        except Exception as e:
            # Log only specific errors if needed, avoid spamming console for timeouts
            if "timeout" not in str(e).lower():
                print(f"Warning: Failed to get Tello state '{name}': {e}")
            return None # Return None on failure


    def _command_processing_loop(self):
        """Processes commands from the queue sequentially."""
        print("Command processing thread started.")
        while not self._stop_event.is_set():
            try:
                # Wait for a command, with timeout to allow checking stop_event
                command = self.command_queue.get(timeout=config.UPDATE_INTERVAL_COMMAND_QUEUE)

                if command is None: # Sentinel value to exit
                    print("Command thread received exit signal.")
                    break

                if self._is_connected and self.tello:
                    print(f"Sending command: {command}")
                    try:
                        # Use send_command_with_return for blocking behavior until 'ok' or error
                        response = self.tello.send_command_with_return(command)
                        print(f"Command '{command}' response: {response}")
                        if response != 'ok':
                             # Optionally handle specific non-'ok' responses
                             print(f"Warning: Command '{command}' returned '{response}'")
                             # Maybe update GUI status based on error?
                    except Exception as e:
                        print(f"Command '{command}' failed: {e}")
                        # Optionally update GUI status
                        # self.root.after(0, self._update_status, f"Cmd Error: {command}")
                else:
                    print(f"Skipping command '{command}': Not connected.")

                self.command_queue.task_done() # Mark command as processed

            except queue.Empty:
                # Timeout reached, loop continues to check _stop_event
                continue
            except Exception as e:
                print(f"Error in command processing thread: {e}")
                time.sleep(1) # Wait after unexpected error

        print("Command processing thread finished.")


    def _plot_update_loop(self):
        """Periodically schedules the 3D plot update on the main thread."""
        print("Plot update thread started.")
        while not self._stop_event.is_set():
            try:
                if self._is_connected and hasattr(self, 'plot_manager'):
                    # Get current command-based trajectory
                    cmd_traj = self.trajectory[:] # Get a copy

                    # Get current VO trajectory (if enabled)
                    current_vo_traj = self.vo_trajectory[:] if self.vo_enabled.get() else None

                    # Schedule the update in the main GUI thread
                    # *** PASS ALL ARGUMENTS POSITIONALLY ***
                    self.root.after(0, self.plot_manager.update_plot,
                                    self.estimated_x,       # Arg 1 (x_cmd)
                                    self.estimated_y,       # Arg 2 (y_cmd)
                                    self.estimated_z,       # Arg 3 (z_cmd)
                                    self.estimated_yaw,     # Arg 4 (yaw_deg_cmd)
                                    cmd_traj,               # Arg 5 (trajectory_cmd)
                                    current_vo_traj,        # Arg 6 (vo_trajectory)
                                    self.vo_enabled.get()   # Arg 7 (vo_enabled)
                                   )

                time.sleep(config.UPDATE_INTERVAL_PLOT)
            except Exception as e:
                # Adding traceback for better debugging if other errors occur
                import traceback
                print(f"Error in plot update thread: {e}")
                traceback.print_exc()
                time.sleep(1)
        print("Plot update thread finished.")


    # --- GUI Update Callbacks (run in main thread via root.after) ---

    def _update_video_label_image(self, imgtk):
        """Updates the video label with a new image."""
        if self.video_label.winfo_exists():
            self.video_label.config(image=imgtk, text="") # Display image, clear text
            self.video_label.image = imgtk # Keep a reference! Important.
        else:
            print("Video label does not exist for update.") # Debugging

    def _update_video_label_text(self, text):
         """Updates the video label with text (e.g., for errors or status)."""
         if self.video_label.winfo_exists():
              self.video_label.config(text=text, image="") # Display text, clear image
              self.video_label.image = None # Clear reference


    def _update_state_labels(self):
        """Updates the state labels in the GUI."""
        if not self._is_connected or not self.root.winfo_exists(): return
        try:
            self.battery_var.set(f"{self.current_bat}%")
            self.height_var.set(f"{self.current_h} cm")
            self.tof_var.set(f"{self.current_tof} cm")
            self.attitude_var.set(f"{self.current_pitch}°, {self.current_roll}°, {self.current_yaw}°")
            self.speed_var.set(f"{self.current_vgx}, {self.current_vgy}, {self.current_vgz}")
            self.time_var.set(f"{self.current_time}s")
            self._update_odometry_display() # Keep odometry updated too
        except tk.TclError as e:
             # Can happen if GUI is closing during update
             print(f"Minor GUI update error (state labels): {e}")

    def _update_odometry_display(self):
        """Updates the estimated odometry label."""
        if hasattr(self, 'odom_var') and self.odom_var and self.root.winfo_exists():
            odom_text = (f"{self.estimated_x:.1f}, {self.estimated_y:.1f}, "
                         f"{self.estimated_z:.1f}, {self.estimated_yaw:.1f}°")
            try:
                self.odom_var.set(odom_text)
            except tk.TclError as e:
                print(f"Minor GUI update error (odometry label): {e}")

    def _update_status(self, message):
         """Updates the main status label."""
         if hasattr(self, 'status_var') and self.status_var and self.root.winfo_exists():
             try:
                 self.status_var.set(f"Status: {message}")
             except tk.TclError as e:
                 print(f"Minor GUI update error (status label): {e}")

    def _reset_gui_state(self):
         """Resets GUI elements to the disconnected state."""
         if not self.root.winfo_exists(): return
         try:
            self.connect_button.config(text="Connect", command=self.connect_tello)
            self.takeoff_button.config(state=tk.DISABLED)
            self.land_button.config(state=tk.DISABLED) # Disable land when disconnected

            self.battery_var.set("N/A")
            self.height_var.set("N/A cm")
            self.tof_var.set("N/A cm")
            self.attitude_var.set("N/A")
            self.speed_var.set("N/A")
            self.time_var.set("N/A")
            self.odom_var.set("0.0, 0.0, 0.0, 0.0°") # Reset odom display

            self._update_video_label_text("Video Feed Disconnected") # Reset video label
         except tk.TclError as e:
              print(f"Minor GUI reset error: {e}")


    # --- Window Closing ---

    def on_closing(self):
        """Handles the window close event, including closing MediaPipe."""
        print("Close button clicked.")
        # Close MediaPipe Detector if it exists
        if hasattr(self, 'object_detector') and self.object_detector:
            try:
                # The Tasks API objects don't always have an explicit close().
                # Often releasing the reference is enough. Let's remove the close call for now.
                print("Releasing MediaPipe detector reference...")
                self.object_detector = None # Allow garbage collection
            except Exception as e:
                print(f"Error during MediaPipe detector cleanup (ignored): {e}")

        if self._is_connected:
            if messagebox.askokcancel("Quit", "Do you want to quit?\nThis will attempt to land the drone and disconnect."):
                self.disconnect_tello()
                self.root.destroy()
            else:
                 return
        else:
            self.root.destroy()

        print("Application Root Destroyed.")
