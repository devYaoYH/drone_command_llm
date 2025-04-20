# gui.py
"""Defines the main Tello GUI Application class with Voice & Agent control."""

import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import cv2
from PIL import Image, ImageTk
# Removed direct Tello import: from djitellopy import Tello
import math
import queue
import numpy as np
import os # For file paths if needed
import json

# --- Imports from app.py ---
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from openai import OpenAI
# --- End imports from app.py ---

# --- Imports for Agent/Vision System ---
from agent.agent import TelloDroneVisionController  # Now using absolute import
import config  # Import shared configuration
from plotter import PlotManager  # Import the plot manager
from visual_odometry import VisualOdometry  # Keep VO for visualization/comparison
import camera_calibration  # Keep for VO camera parameters
# --- End imports for Agent/Vision System ---

# --- MediaPipe Imports ---
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks import python as mp_python_tasks
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

MP_MODEL_FILENAME = 'efficientdet_lite0.tflite' # Define if using MP
MP_MODEL_PATH = os.path.join(os.path.dirname(__file__), MP_MODEL_FILENAME)
# --- End MediaPipe Imports ---


# ---- Global Configs (Moved from app.py or defined here) ----
WHISPER_SIZE = "base" # Or tiny, small, medium
NUM_CORES = os.cpu_count() // 2 # Use half cores for Whisper to leave resources
WAKE_WORD = "hey drone" # Or "tello", "drone", etc.
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_LISTEN = 3 # Seconds for wake word / cancel check
AUDIO_DURATION_COMMAND = 5 # Seconds for command listening


class TelloGUIApp:
    """Main application class for Tello control GUI with Agent/Voice control."""

    def __init__(self, root):
        self.root = root
        self.root.title("Tello Vision Agent Controller")

        # --- Agent & Drone State ---
        self.controller = TelloDroneVisionController(gui_instance=self) # Instantiate Agent, pass self
        # self._is_connected = False # Agent handles connection status via self.controller.is_connected()
        self._is_flying = False # Track flight state (takeoff/land commands)
        self._stop_event = threading.Event() # For stopping background threads
        self.latest_cv_frame = None # Store latest frame for agent/display

        # --- Voice Control State ---
        self.whisper_model = None
        self.openai_client = None
        self.voice_listening_thread = None
        self._voice_thread_stop_event = threading.Event() # Separate stop for voice loop
        self.is_listening_for_command = False
        self.current_agent_task_thread = None

        # --- GUI State Variables (Many initialized here) ---
        self.status_var = tk.StringVar(value="Status: Disconnected")
        self.battery_var = tk.StringVar(value="Battery: N/A")
        self.height_var = tk.StringVar(value="Height AGL: N/A cm")
        self.tof_var = tk.StringVar(value="TOF: N/A cm")
        self.attitude_var = tk.StringVar(value="Attitude (P,R,Y): N/A")
        self.speed_var = tk.StringVar(value="Speed (X,Y,Z): N/A")
        self.time_var = tk.StringVar(value="Flight Time: N/A")
        self.odom_var = tk.StringVar(value="Cmd Odom: 0.0, 0.0, 0.0, 0.0°") # Keep cmd odom?
        self.vo_status_var = tk.StringVar(value="VO Status: Disabled")
        self.object_detection_status_var = tk.StringVar(value="Obj Detect: Disabled") # If keeping MP
        self.voice_status_var = tk.StringVar(value="Voice: Idle")
        self.last_command_var = tk.StringVar(value="Last Cmd: N/A")
        self.agent_status_var = tk.StringVar(value="Agent: Idle")

        # --- Background Threads (GUI Related) ---
        self.state_thread = None
        self.video_thread = None
        self.plot_thread = None
        # Command queue might be less relevant if agent controls steps, but keep for now
        # self.command_queue = queue.Queue()
        # self.command_thread = None

        # --- Odometry & Visualization Components (Keep for comparison/display) ---
        self.estimated_x = 0.0 # Command-based estimate
        self.estimated_y = 0.0
        self.estimated_z = 0.0
        self.estimated_yaw = 0.0
        self.trajectory = []
        self.vo_enabled = tk.BooleanVar(value=config.VO_ENABLE_BY_DEFAULT) # VO can still run for vis
        self.vo_processor = None
        self.vo_trajectory = []
        # Optional: Feature/Object Detection for Display
        self.feature_detection_enabled = tk.BooleanVar(value=config.FEATURE_DETECTION_ENABLE_BY_DEFAULT)
        self.orb_detector = None
        self.object_detection_enabled = tk.BooleanVar(value=False) # If keeping MP display
        self.orb_detector = None
        self.object_detector = None # MediaPipe detector

        # --- Initialization ---
        self._initialize_voice_components() # Init Whisper, OpenAI
        self._initialize_visualization_components() # Init VO, Features, MP (Optional)

        # --- GUI Setup ---
        self._setup_gui_layout()
        self._create_gui_widgets() # Create all labels, buttons, etc.

        # Plotter Setup
        self.plot_manager = PlotManager(self.plot_frame)

        # Graceful Shutdown
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- Initialization Methods ---

    def _initialize_voice_components(self):
        """Initializes Whisper and OpenAI TTS."""
        print("Initializing Voice Components...")
        try:
            print(f"Loading Whisper model '{WHISPER_SIZE}'...")
            self.whisper_model = WhisperModel(
                WHISPER_SIZE,
                device="cpu",
                compute_type="int8", # Use int8 for CPU efficiency
                cpu_threads=NUM_CORES,
                num_workers=NUM_CORES
            )
            print("Whisper model loaded.")
        except Exception as e:
            print(f"ERROR: Failed to load Whisper model: {e}")
            messagebox.showerror("Whisper Error", f"Failed to load Whisper model: {e}")
            self.whisper_model = None

        try:
            # Ensure API key is set (can check os.getenv directly)
            if not os.getenv("OPENAI_API_KEY"):
                 print("Warning: OPENAI_API_KEY environment variable not set. TTS will fail.")
                 # Optionally disable speak functionality or show warning
            self.openai_client = OpenAI() # Key is read from env var automatically
            print("OpenAI client initialized.")
        except Exception as e:
            print(f"ERROR: Failed to initialize OpenAI client: {e}")
            messagebox.showerror("OpenAI Error", f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None


    def _initialize_visualization_components(self):
        """Initializes VO, Feature Detector, Object Detector for visualization."""
        # Initialize VO (for visualization even if agent controls)
        try:
            self.vo_processor = VisualOdometry(
                cam_matrix=camera_calibration.camera_matrix,
                distortion=camera_calibration.dist_coeffs
            )
            print("Visual Odometry Initialized (for visualization).")
        except Exception as e:
            print(f"ERROR: Failed to initialize Visual Odometry: {e}")
            self.vo_processor = None

        # Initialize ORB (for visualization)
        try:
            self.orb_detector = cv2.ORB_create(nfeatures=config.FEATURE_DETECTION_MAX_FEATURES)
            print("ORB Feature Detector Initialized (for visualization).")
        except Exception as e:
            print(f"ERROR: Failed to initialize ORB Detector: {e}")
            self.orb_detector = None

        # Optional: Initialize MediaPipe Object Detector (for visualization)
        try:
            if not os.path.exists(MP_MODEL_PATH):
                 print(f"ERROR: MediaPipe model not found at {MP_MODEL_PATH}")
                 messagebox.showerror("MediaPipe Error", f"Model file not found:\n{MP_MODEL_PATH}\nObject Detection disabled.")
                 self.object_detector = None
            else:
                print(f"Loading MediaPipe model from: {MP_MODEL_PATH}")
                # Use the Tasks API - more modern
                base_options = mp_tasks.BaseOptions(model_asset_path=MP_MODEL_PATH)
                options = mp_vision.ObjectDetectorOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.IMAGE, # Process frame by frame
                    score_threshold=0.4, # Adjust sensitivity
                    max_results=5 # Limit number of detections
                )
                self.object_detector = mp_vision.ObjectDetector.create_from_options(options)
                print("MediaPipe Object Detector Initialized (for visualization).")
        except Exception as e:
             print(f"ERROR: Failed to initialize MediaPipe Object Detection: {e}")
             self.object_detector = None
             messagebox.showerror("MediaPipe Error", f"Failed to initialize MediaPipe: {e}")
        # --- End MediaPipe Initialization ---


    # --- GUI Setup Methods ---

    def _setup_gui_layout(self):
        """Creates the main frames and configures grid layout."""
        # ... (Layout setup remains the same as before) ...
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

        # Create plot frame for PlotManager
        self.plot_frame = ttk.LabelFrame(self.right_bottom_frame, text="3D Digital Twin", padding="5")
        self.plot_frame.grid(row=0, column=0, sticky="nsew")
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

    def _create_gui_widgets(self):
        """Creates and places widgets, replacing manual controls with status."""

        # --- State Frame Widgets (Enhanced) ---
        state_frame = ttk.LabelFrame(self.left_frame, text="System Status", padding="10")
        state_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Row counter for state frame
        row_idx = 0

        # Basic Drone State
        ttk.Label(state_frame, text="Status:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(state_frame, textvariable=self.status_var).grid(row=row_idx, column=1, sticky="w", padx=5, pady=1); row_idx += 1
        ttk.Label(state_frame, text="Battery:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(state_frame, textvariable=self.battery_var).grid(row=row_idx, column=1, sticky="w", padx=5, pady=1); row_idx += 1
        ttk.Label(state_frame, text="TOF:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(state_frame, textvariable=self.tof_var).grid(row=row_idx, column=1, sticky="w", padx=5, pady=1); row_idx += 1
        ttk.Label(state_frame, text="Attitude:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(state_frame, textvariable=self.attitude_var).grid(row=row_idx, column=1, sticky="w", padx=5, pady=1); row_idx += 1

        # Separator
        ttk.Separator(state_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky='ew', pady=5); row_idx += 1

        # Voice Status
        ttk.Label(state_frame, text="Voice:", font=("Arial", 9, "bold")).grid(row=row_idx, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(state_frame, textvariable=self.voice_status_var).grid(row=row_idx, column=1, sticky="w", padx=5, pady=1); row_idx += 1
        ttk.Label(state_frame, text="Last Cmd:", font=("Arial", 9, "bold")).grid(row=row_idx, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(state_frame, textvariable=self.last_command_var, wraplength=250).grid(row=row_idx, column=1, sticky="w", padx=5, pady=1); row_idx += 1 # Wraplength for long commands

        # Agent Status
        ttk.Label(state_frame, text="Agent:", font=("Arial", 9, "bold")).grid(row=row_idx, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(state_frame, textvariable=self.agent_status_var).grid(row=row_idx, column=1, sticky="w", padx=5, pady=1); row_idx += 1

        # Separator
        ttk.Separator(state_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky='ew', pady=5); row_idx += 1

        # Visualization Status (Optional)
        ttk.Label(state_frame, text="Vis Status:", font=("Arial", 9, "bold")).grid(row=row_idx, column=0, sticky="w", padx=5, pady=1)
        ttk.Label(state_frame, textvariable=self.vo_status_var, font=("Arial", 9, "italic")).grid(row=row_idx, column=1, sticky="w", padx=5, pady=1); row_idx += 1
        # Add object detection status if kept
        ttk.Label(state_frame, textvariable=self.object_detection_status_var, font=("Arial", 9, "italic")).grid(row=row_idx, column=1, sticky="w", padx=5, pady=1); row_idx += 1

        # --- Control Frame Widgets (Modified) ---
        control_frame = ttk.LabelFrame(self.left_frame, text="System Control", padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)

        # Row 1: Connect, Takeoff (Task), Land (Task)
        self.connect_button = ttk.Button(control_frame, text="Connect Agent", command=self.connect_agent_system) # Changed command
        self.connect_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        # Takeoff/Land now trigger agent tasks
        self.takeoff_button = ttk.Button(control_frame, text="Task: Take Off", command=lambda: self._trigger_agent_task("takeoff"), state=tk.DISABLED)
        self.takeoff_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.land_button = ttk.Button(control_frame, text="Task: Land", command=lambda: self._trigger_agent_task("land"), state=tk.DISABLED)
        self.land_button.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        # Row 2: Emergency Stop Button (Keep this direct control)
        self.emergency_button = ttk.Button(control_frame, text="! EMERGENCY STOP !", command=self.emergency_stop) # Changed command
        self.emergency_button.grid(row=2, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

        # Row 3: Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=3, column=0, columnspan=3, sticky='ew', pady=5)

        # Row 4, 5, 6: Visualization Toggles (Optional)
        self.feature_checkbox = tk.Checkbutton(control_frame, text="Show Features (ORB)",
                                               variable=self.feature_detection_enabled)
                                               # command=self._on_feature_toggle) # Optional callback if needed
        self.feature_checkbox.grid(row=4, column=0, columnspan=3, sticky='w', padx=5, pady=2)
        if not self.orb_detector: self.feature_checkbox.config(state=tk.DISABLED)

        self.vo_checkbox = tk.Checkbutton(control_frame, text="Show Visual Odometry (VO)",
                                          variable=self.vo_enabled)
                                          # command=self._on_vo_toggle) # Optional callback
        self.vo_checkbox.grid(row=5, column=0, columnspan=3, sticky='w', padx=5, pady=2)
        if not self.vo_processor: self.vo_checkbox.config(state=tk.DISABLED)

        
        # Row 6 (or adjust row index as needed)
        self.obj_detect_checkbox = tk.Checkbutton(control_frame, text="Show Object Detection (MP)",
                                                  variable=self.object_detection_enabled)
        self.obj_detect_checkbox.grid(row=6, column=0, columnspan=3, sticky='w', padx=5, pady=2)
        if not self.object_detector: # Disable if init failed
             self.obj_detect_checkbox.config(state=tk.DISABLED)
        # --- End MP Checkbox ---

        # --- Video Frame Widgets ---
        video_frame = ttk.LabelFrame(self.right_top_frame, text="Video Stream", padding="5")
        video_frame.grid(row=0, column=0, sticky="nsew")
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        self.video_label = ttk.Label(video_frame, text="Video Feed Disconnected", anchor=tk.CENTER)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # --- 3D Plot Frame (Populated by PlotManager later) ---
        # Plot frame itself is created in _setup_gui_layout


    # --- Agent/Drone Connection ---

    def connect_agent_system(self):
        """Connects the agent, which handles drone/server connection."""
        if self.controller.is_connected():
             print("System already connected. Disconnect first.")
             # Optionally change button to "Disconnect" here
             # self.connect_button.config(text="Disconnect", command=self.disconnect_agent_system)
             return

        self.update_status("Connecting Agent...")
        self.root.update_idletasks()

        # Try connecting the controller (runs in main thread, assumes connect is quick)
        connect_success = self.controller.connect()

        if connect_success:
            self.update_status("Agent Connected")
            # Update button state
            self.connect_button.config(text="Disconnect", command=self.disconnect_agent_system)
            self.takeoff_button.config(state=tk.NORMAL)
            self.land_button.config(state=tk.NORMAL)
            self.emergency_button.config(state=tk.NORMAL) # Enable emergency

            # Reset visualization states
            if self.vo_processor: self.vo_processor.reset()
            self.vo_trajectory = [(0.0, 0.0, 0.0)]
            self.estimated_x = self.estimated_y = self.estimated_z = self.estimated_yaw = 0.0
            self.trajectory = [(0.0, 0.0, 0.0)]
            if hasattr(self, 'plot_manager') and self.plot_manager:
                 self.plot_manager.reset_plot()

            # Start GUI background threads
            self._start_gui_threads()

            # Start the voice listening loop in its own thread
            self._start_voice_listening_thread()

            # Optional: Initial greeting
            self._speak("Drone connected and ready.")

        else:
            self.update_status("Agent Connection Failed")
            messagebox.showerror("Connection Error", "Failed to connect Agent.\nCheck Drone/Server/Network.")
            # Ensure button is back to "Connect" state
            self.connect_button.config(text="Connect Agent", command=self.connect_agent_system)
            self.takeoff_button.config(state=tk.DISABLED)
            self.land_button.config(state=tk.DISABLED)
            self.emergency_button.config(state=tk.DISABLED)


    def disconnect_agent_system(self):
        """Disconnects the agent, stops threads, and lands the drone."""
        if not self.controller.is_connected():
            print("System already disconnected.")
            return

        print("Disconnecting Agent System...")
        self.update_status("Disconnecting...")

        # 1. Stop Voice Listening Thread
        if self.voice_listening_thread and self.voice_listening_thread.is_alive():
            print("Stopping voice loop...")
            self._voice_thread_stop_event.set()
            # Don't join here, might block GUI

        # 2. Cancel any running Agent Task
        if hasattr(self.controller, 'cancel_task'):
             print("Requesting agent task cancellation...")
             self.controller.cancel_task()
             # Wait briefly for task thread to potentially acknowledge
             if self.current_agent_task_thread and self.current_agent_task_thread.is_alive():
                  self.current_agent_task_thread.join(timeout=1.0) # Short wait

        # 3. Land the Drone (via agent or directly if needed)
        print("Ensuring drone is landed...")
        try:
            # Prefer agent's landing if possible, might already be landing due to cancel
             if self._is_flying: # Check GUI's flight state knowledge
                 self.controller.land_drone_immediately()
                 # Add a small delay for landing command to process
                 time.sleep(3) # Adjust as needed
                 self._is_flying = self.controller.drone.is_flying() # Assume landed
        except Exception as e:
             print(f"Error during agent landing: {e}")
             # Fallback: Try direct emergency via API if accessible
             try:
                  if self.controller.drone and self.controller.drone.api:
                       self.controller.drone.api.emergency()
                       time.sleep(1)
                       self._is_flying = self.controller.drone.is_flying() # Assume landed
             except Exception as emer_e:
                  print(f"Emergency land fallback failed: {emer_e}")

        # 4. Stop GUI Background Threads
        print("Stopping GUI threads...")
        self._stop_event.set()
        # No command thread to stop in this model

        # 5. Wait for Threads to Finish (with timeouts)
        threads_to_join = [
            self.state_thread, self.video_thread, self.plot_thread,
            self.voice_listening_thread, self.current_agent_task_thread
        ]
        for t in threads_to_join:
             if t and t.is_alive():
                  t.join(timeout=1.5)
        print("Threads joined.")

        # 6. Tello object disconnect is handled internally by agent/drone_api potentially
        # We might not need self.controller.disconnect() unless it releases resources
        self.controller.connected = False # Manually update status if agent doesn't

        # 7. Reset GUI State
        self._reset_gui_state()
        print("Agent System Disconnected.")

    def emergency_stop(self):
        """Direct emergency command, bypassing agent logic."""
        print("!!! EMERGENCY STOP Initiated !!!")
        # Cancel agent task immediately
        if hasattr(self.controller, 'cancel_task'):
             self.controller.cancel_task()
        # Send direct emergency command
        if self.controller.is_connected() and self.controller.drone.api:
             try:
                  self.controller.drone.api.emergency()
                  time.sleep(1)
                  self._is_flying = self.controller.drone.is_flying() # Assume stopped
                  self.update_status("EMERGENCY STOPPED")
                  self.update_agent_status("EMERGENCY")
                  # Force UI update?
                  self.root.after(100, self._reset_gui_on_disconnect) # Reset buttons etc after short delay
             except Exception as e:
                  print(f"Failed to send emergency command: {e}")
                  self.update_status("Emergency Failed!")
        else:
             print("Cannot send emergency: Drone not connected.")

    def update_command_odometry(self, command_type, value):
        """Updates estimated position based on a completed command."""
        try:
            # Use the most recent Tello yaw as the reference for calculation
            # This helps reduce drift in the command estimate's yaw component
            # current_yaw_deg = self.current_yaw # Get yaw from state update
            # Or use the estimated yaw if you prefer pure command odometry
            current_yaw_deg = self.estimated_yaw
            current_yaw_rad = math.radians(current_yaw_deg)

            dx, dy, dz, dyaw = 0.0, 0.0, 0.0, 0.0
            value = float(value) # Ensure value is a float

            if command_type in ["forward", "move_forward"]:
                dx = value * math.cos(current_yaw_rad)
                dy = value * math.sin(current_yaw_rad) # Positive Y is Left
            elif command_type in ["back", "move_back", "backward"]:
                dx = -value * math.cos(current_yaw_rad)
                dy = -value * math.sin(current_yaw_rad)
            elif command_type in ["left", "move_left"]:
                # Move perpendicular (positive Y direction relative to drone)
                dx = value * math.cos(current_yaw_rad + math.pi/2) # cos(a+pi/2) = -sin(a)
                dy = value * math.sin(current_yaw_rad + math.pi/2) # sin(a+pi/2) = cos(a)
                # dx = -value * math.sin(current_yaw_rad)
                # dy = value * math.cos(current_yaw_rad)
            elif command_type in ["right", "move_right"]:
                # Move perpendicular (negative Y direction relative to drone)
                dx = value * math.cos(current_yaw_rad - math.pi/2) # cos(a-pi/2) = sin(a)
                dy = value * math.sin(current_yaw_rad - math.pi/2) # sin(a-pi/2) = -cos(a)
                # dx = value * math.sin(current_yaw_rad)
                # dy = -value * math.cos(current_yaw_rad)
            elif command_type in ["up", "move_up"]:
                # dz = value # Z is handled by TOF in state update loop
                pass
            elif command_type in ["down", "move_down"]:
                # dz = -value # Z is handled by TOF in state update loop
                pass
            elif command_type in ["cw", "rotate_clockwise"]:
                dyaw = -value # Decreasing angle for clockwise
            elif command_type in ["ccw", "rotate_counter_clockwise"]:
                dyaw = value # Increasing angle for counter-clockwise

            # Update estimates
            self.estimated_x += dx
            self.estimated_y += dy
            # self.estimated_z += dz # Keep Z updated from TOF in _state_update_loop
            self.estimated_yaw += dyaw
            # Normalize yaw (optional but good practice)
            self.estimated_yaw = (self.estimated_yaw + 180) % 360 - 180

            # --- Append to trajectory ---
            # Use the TOF-updated Z
            current_z = self.estimated_z
            new_point = (self.estimated_x, self.estimated_y, current_z)
            self.trajectory.append(new_point)

            # Limit trajectory length
            max_len = config.PLOT_TRAJECTORY_LENGTH
            self.trajectory = self.trajectory[-max_len:]

            print(f"  Cmd Odom Update: Cmd='{command_type} {value}', dX={dx:.1f}, dY={dy:.1f}, dYaw={dyaw:.1f} -> Pos=({self.estimated_x:.1f},{self.estimated_y:.1f},{current_z:.1f}) Yaw={self.estimated_yaw:.1f}")

        except Exception as e:
            print(f"Error updating command odometry: {e}")


    def _reset_gui_state(self):
        """Resets GUI elements to the disconnected state."""
        self.update_status("Disconnected")
        self.battery_var.set("N/A")
        self.height_var.set("N/A cm")
        self.tof_var.set("N/A cm")
        self.attitude_var.set("N/A")
        self.speed_var.set("N/A")
        self.time_var.set("N/A")
        self.odom_var.set("0.0, 0.0, 0.0, 0.0°")
        self.vo_status_var.set("VO Status: Disabled")
        self.voice_status_var.set("Voice: Idle")
        self.last_command_var.set("Last Cmd: N/A")
        self.agent_status_var.set("Agent: Idle")
        # If using object detection display:
        # self.object_detection_status_var.set("Obj Detect: Disabled")

        self.connect_button.config(text="Connect Agent", command=self.connect_agent_system)
        self.takeoff_button.config(state=tk.DISABLED)
        self.land_button.config(state=tk.DISABLED)
        self.emergency_button.config(state=tk.DISABLED)

        # Clear video label
        if hasattr(self, 'video_label') and self.video_label.winfo_exists():
            self._update_video_label_text("Video Feed Disconnected")

        # Reset plot
        if hasattr(self, 'plot_manager') and self.plot_manager:
             self.plot_manager.reset_plot()

        # Reset state flags
        self._is_flying = self.controller.drone.is_flying()
        self._stop_event.clear() # Clear main stop event
        self._voice_thread_stop_event.clear() # Clear voice stop event


    # --- Background Thread Management ---

    def _start_gui_threads(self):
        """Starts the GUI background threads for state, video, plot."""
        self._stop_event.clear() # Ensure stop event is clear

        if not self.state_thread or not self.state_thread.is_alive():
            self.state_thread = threading.Thread(target=self._state_update_loop, daemon=True)
            self.state_thread.start()

        if not self.video_thread or not self.video_thread.is_alive():
            self.video_thread = threading.Thread(target=self._video_update_loop, daemon=True)
            self.video_thread.start()

        if not self.plot_thread or not self.plot_thread.is_alive():
            self.plot_thread = threading.Thread(target=self._plot_update_loop, daemon=True)
            self.plot_thread.start()

    def _start_voice_listening_thread(self):
        """Starts the voice command listening loop in a background thread."""
        if not self.whisper_model or not self.openai_client:
             messagebox.showwarning("Voice Init Error", "Whisper or OpenAI not initialized. Voice control disabled.")
             self.update_voice_status("Voice Disabled (Init Failed)")
             return

        if self.voice_listening_thread and self.voice_listening_thread.is_alive():
            print("Voice listening thread already running.")
            return

        print("Starting voice listening thread...")
        self._voice_thread_stop_event.clear() # Ensure stop is clear before starting
        self.voice_listening_thread = threading.Thread(target=self._voice_listening_loop, daemon=True)
        self.voice_listening_thread.start()
        self.update_voice_status(f"Listening for '{WAKE_WORD}'...")


    # --- Background Thread Loops (State, Video, Plot - Adapted) ---

    def _state_update_loop(self):
        """Periodically fetches Tello state via the controller and updates GUI."""
        print("State update thread started.")
        while not self._stop_event.is_set():
            # Check connection via controller
            if self.controller.is_connected():
                try:
                    # Access state via controller.drone.api
                    api = self.controller.drone.api
                    bat = api.get_battery()
                    tof = api.get_distance_tof() # Use TOF for altitude sensing
                    h = api.get_height() # AGL estimate
                    pitch = api.get_pitch()
                    roll = api.get_roll()
                    yaw = api.get_yaw() # Tello's reported yaw
                    vgx = api.get_speed_x()
                    vgy = api.get_speed_y()
                    vgz = api.get_speed_z()
                    flight_time = api.get_flight_time()

                    # --- Update Internal State (including command odom if kept) ---
                    self.current_bat = bat
                    self.current_h = h
                    self.current_tof = tof
                    self.current_pitch = pitch
                    self.current_roll = roll
                    self.current_yaw = yaw # Tello's yaw
                    self.current_vgx = vgx
                    self.current_vgy = vgy
                    self.current_vgz = vgz
                    self.current_time = flight_time

                    # Update command-based Z estimate using TOF
                    # Only update Z if TOF reading is plausible (e.g., > 0)
                    if tof is not None and tof > 0:
                         self.estimated_z = tof # Update Z with TOF reading

                    # Schedule GUI update
                    self.root.after(0, self._update_state_labels)

                except Exception as e:
                    # Handle potential errors if API calls fail during connection loss etc.
                    # print(f"Error in state update loop: {e}") # Can be noisy
                    # Check if disconnected mid-loop
                    if not self.controller.is_connected():
                         print("State thread: Controller disconnected, stopping loop.")
                         break
                    time.sleep(0.5) # Wait longer after error
            else:
                 # No longer connected, exit thread cleanly
                 break

            time.sleep(config.UPDATE_INTERVAL_STATE)
        print("State update thread finished.")


    def _video_update_loop(self):
        """Handles video feed, stores latest frame, runs visualization."""
        print("Video update thread started.")
        frame_read = None
        last_detection_send_time = 0
        detection_send_interval = 0.2 # Max send rate (5 times per second)
        
        while not self._stop_event.is_set():
            try:
                if not self.controller.is_connected():
                    break # Exit if controller disconnects
                
                # Get video reader if not already obtained
                if frame_read is None:
                    frame_read = self.controller.get_video_reader()
                    if frame_read is None:
                        print("Video Thread: Waiting for frame reader...")
                        time.sleep(0.1) # Wait if no reader yet
                        continue
                    print("Video Thread: Frame reader obtained.")
                
                # Try to get a frame
                frame = frame_read.frame
                if frame is None:
                    print("Video Thread: No frame available")
                    time.sleep(0.1)
                    continue
                
                # Make a copy of the frame for processing
                display_frame = frame.copy()
                self.latest_cv_frame = display_frame
                processed_detections_for_server = None

                if frame_read.stopped:
                    print("Video thread stopping: Tello video source stopped.")
                    break

                # --- Process frame for visualization ---
                # 1. Run VO (if enabled) - Use internal process_frame for visualization
                if self.vo_enabled.get() and self.vo_processor:
                     try:
                          # VO process frame purely for visualization output
                          # Pass None for altitude if not critical for vis-only VO run
                          _, _, vo_output_frame, vo_quality = self.vo_processor.process_frame(display_frame.copy(), altitude=None)
                          display_frame = vo_output_frame # Update frame with VO overlays
                          # --- FIX: Get the updated trajectory ---
                          current_vo_traj_full = self.vo_processor.get_trajectory()
                          # Limit length for performance/memory
                          max_len = config.VO_TRAJECTORY_LENGTH
                          self.vo_trajectory = current_vo_traj_full[-max_len:]
                          # --- END FIX ---
                          
                          # Update VO status label in GUI
                          self.root.after(0, self.vo_status_var.set, f"VO Status: Running (Qual: {vo_quality:.2f})")
                     except Exception as vo_e:
                          # print(f"VO visualization error: {vo_e}") # Can be noisy
                          self.root.after(0, self.vo_status_var.set, f"VO Status: Error")


                # 2. Run Feature Detection (if enabled and VO didn't run)
                if self.feature_detection_enabled.get() and not self.vo_enabled.get() and self.orb_detector:
                     # Draw ORB features... (code from previous versions)
                      try:
                           gray_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                           keypoints = self.orb_detector.detect(gray_display, None)
                           display_frame = cv2.drawKeypoints(display_frame, keypoints, None, color=(0, 255, 0), flags=0)
                      except Exception as orb_e: print(f"ORB drawing error: {orb_e}")


                # --- 3. Run MediaPipe Object Detection (if enabled) ---
                if self.object_detection_enabled.get() and self.object_detector:
                    try:
                        # Convert BGR frame to RGB for MediaPipe
                        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                        # Perform detection
                        detection_result = self.object_detector.detect(mp_image)

                        # Prepare data for server and draw on local frame
                        processed_detections_for_server = self._process_and_draw_mp_detections(
                            display_frame, # Draw onto this frame (BGR)
                            detection_result
                        )

                        # Update local GUI status
                        num_detections = len(detection_result.detections)
                        self.root.after(0, self.object_detection_status_var.set, f"Obj Detect: Running ({num_detections} found)")

                    except Exception as mp_e:
                        print(f"Error during MediaPipe detection/drawing: {mp_e}")
                        self.root.after(0, self.object_detection_status_var.set, "Obj Detect: Error")
                        processed_detections_for_server = None # Ensure it's reset on error
                else:
                     # If detection is disabled, ensure status reflects that
                     self.root.after(0, self.object_detection_status_var.set, "Obj Detect: Disabled")


                # --- Send Detections to Server (if any and interval passed) ---
                current_time = time.time()
                if (processed_detections_for_server is not None and
                        current_time - last_detection_send_time > detection_send_interval):
                    try:
                        # Use the controller's method to send data
                        if self.controller.is_connected():
                            self.controller.post_data_to_server(
                                "/update_detections",
                                {"detections": processed_detections_for_server}
                            )
                            last_detection_send_time = current_time
                            # print(f"Sent {len(processed_detections_for_server)} detections to server.") # Debug
                    except Exception as send_err:
                        print(f"Error sending detections to server: {send_err}")


                # --- Final Display Preparation ---
                h, w, _ = display_frame.shape
                ratio = min(config.VIDEO_MAX_WIDTH / w, config.VIDEO_MAX_HEIGHT / h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                frame_resized = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                # Schedule GUI update
                self.root.after(0, self._update_video_label_image, imgtk)

            except Exception as e:
                print(f"Error in video loop: {e}")
                import traceback
                traceback.print_exc()
                self.root.after(0, self._update_video_label_text, "Video Error")
                self.latest_cv_frame = None # Clear frame on error
                frame_read = None # Reset frame reader on error
                time.sleep(1) # Pause after error

            # Control loop rate
            time.sleep(config.UPDATE_INTERVAL_VIDEO)

        print("Video update thread finished.")
        # Clear label when thread stops
        self.root.after(0, self._update_video_label_text, "Video Feed Stopped")
        self.latest_cv_frame = None
    
    def _process_and_draw_mp_detections(self, image_bgr, detection_result):
        """
        Draws MediaPipe detections on the image and extracts data for the server.

        Args:
            image_bgr: The OpenCV BGR image to draw on.
            detection_result: The result from MediaPipe ObjectDetector.detect().

        Returns:
            list: A list of dictionaries, each containing 'box' (normalized),
                  'label', and 'score' for detected objects. Returns None if no detections.
        """
        detection_list_for_server = []
        if not detection_result or not detection_result.detections:
            return None # No detections to process

        height, width, _ = image_bgr.shape

        for detection in detection_result.detections:
            bbox = detection.bounding_box
            if not bbox: continue

            # --- Prepare data for server (Normalized Coordinates) ---
            # MediaPipe bounding_box often gives origin_x, origin_y, width, height
            # Convert to xmin, ymin, xmax, ymax and normalize
            xmin = bbox.origin_x / width
            ymin = bbox.origin_y / height
            xmax = (bbox.origin_x + bbox.width) / width
            ymax = (bbox.origin_y + bbox.height) / height

            # Clamp values just in case they slightly exceed bounds
            xmin = max(0.0, min(1.0, xmin))
            ymin = max(0.0, min(1.0, ymin))
            xmax = max(0.0, min(1.0, xmax))
            ymax = max(0.0, min(1.0, ymax))

            category = detection.categories[0] # Get the top category
            label = category.category_name
            score = category.score

            detection_list_for_server.append({
                "box": [xmin, ymin, xmax, ymax], # Normalized [xmin, ymin, xmax, ymax]
                "label": label,
                "score": float(score) # Ensure score is float
            })

            # --- Draw on local BGR image for GUI display ---
            start_point = (bbox.origin_x, bbox.origin_y)
            end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
            cv2.rectangle(image_bgr, start_point, end_point, (0, 255, 0), 2) # Green box

            # Prepare text label
            text = f"{label}: {score:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_origin = (start_point[0], start_point[1] - 6) # Position above box
            # Make sure text is within bounds
            if text_origin[1] < 0: text_origin = (start_point[0], start_point[1] + text_size[1] + 6)

            # Draw background rectangle for text
            cv2.rectangle(image_bgr,
                          (text_origin[0], text_origin[1] - text_size[1] - 2),
                          (text_origin[0] + text_size[0], text_origin[1] + 2),
                          (0, 255, 0), # Green background
                          -1) # Filled

            # Draw text
            cv2.putText(image_bgr, text, text_origin,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, # Black text
                        lineType=cv2.LINE_AA)

        return detection_list_for_server if detection_list_for_server else None

    def _plot_update_loop(self):
        """Periodically schedules the 3D plot update and saves snapshots."""
        print("Plot update thread started.")
        last_save_time = time.time()
        SAVE_INTERVAL = 1  # seconds

        save_counter = 0  # Optional: track how many plots have been saved

        while not self._stop_event.is_set():
            try:
                if self.controller.is_connected() and hasattr(self, 'plot_manager'):
                    # Get copies of trajectories
                    cmd_traj = self.trajectory[:]
                    current_vo_traj = self.vo_trajectory[:] if self.vo_enabled.get() else []

                    self.root.after(0, self.plot_manager.update_plot,
                                    self.estimated_x,
                                    self.estimated_y,
                                    self.estimated_z,
                                    self.estimated_yaw,
                                    cmd_traj,
                                    current_vo_traj,
                                    self.vo_enabled.get())

                    # Periodic saving logic
                    if time.time() - last_save_time >= SAVE_INTERVAL:
                        self.plot_manager.save_plot()
                        last_save_time = time.time()
                        save_counter += 1

                time.sleep(config.UPDATE_INTERVAL_PLOT)

            except Exception as e:
                print(f"Error in plot update thread: {e}")
                time.sleep(1)

        print("Plot update thread finished.")


    # --- Voice Control Methods (Adapted from app.py) ---

    def _listen_for_audio(self, duration, sample_rate=AUDIO_SAMPLE_RATE):
        """Records audio for specified duration."""
        # Use sounddevice for recording
        try:
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait() # Wait for recording to complete
            return audio_data.flatten()
        except Exception as e:
             print(f"Error during audio recording: {e}")
             # Update GUI status?
             self.update_voice_status(f"Error Recording")
             return None

    def _transcribe_audio(self, audio_data, sample_rate=AUDIO_SAMPLE_RATE):
        """Transcribes numpy audio array using Whisper."""
        if audio_data is None or self.whisper_model is None:
             return ""

        # Whisper expects file path or numpy array directly
        try:
            # Transcribe directly from numpy array (more efficient)
            segments, info = self.whisper_model.transcribe(
                 audio_data,
                 language="en", # Optional: specify language
                 # beam_size=5 # Optional: Whisper parameter
            )
            text = ''.join(segment.text for segment in segments).strip().lower()
            # print(f"Whisper Lang: {info.language}, Prob: {info.language_probability:.2f}") # Debug info
            return text
        except Exception as e:
            print(f"Error during transcription: {e}")
            self.update_voice_status("Error Transcribing")
            return ""
    
    def _reset_gui_state(self):
        """Resets GUI elements to the disconnected state."""
        # ... (existing resets) ...
        if hasattr(self, 'object_detection_status_var'):
            self.object_detection_status_var.set("Obj Detect: Disabled")
        if hasattr(self, 'obj_detect_checkbox'):
            self.object_detection_enabled.set(False) # Turn off toggle state
            # Re-enable/disable checkbox based on detector presence
            state = tk.NORMAL if self.object_detector else tk.DISABLED
            self.obj_detect_checkbox.config(state=state)

    def _speak(self, text):
        """Converts text to speech and plays it (blocking)."""
        if not self.openai_client:
             print("TTS client not available.")
             return
        print(f"Speaking: {text}")
        self.update_voice_status(f"Speaking: {text[:30]}...") # Update GUI
        try:
            # --- Blocking TTS Playback ---
            # Note: This will block the thread it's called from (voice loop).
            # For non-blocking, would need separate thread/queue for audio playback.
            with self.openai_client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                response_format="pcm", # Ensure this matches OutputStream format
                input=text
            ) as response:
                # Adapt sample rate if needed (TTS-1 default is 24kHz)
                with sd.OutputStream(samplerate=24000, channels=1, dtype='int16') as stream:
                    for chunk in response.iter_bytes(chunk_size=1024):
                         if self._voice_thread_stop_event.is_set(): break # Allow stopping mid-speech
                         audio_array = np.frombuffer(chunk, dtype=np.int16)
                         stream.write(audio_array)
            # --- End Blocking Playback ---
        except Exception as e:
            print(f"Error during TTS playback: {e}")
            self.update_voice_status("Error Speaking")
        finally:
             # Reset voice status after speaking (or potentially back to listening)
             # This depends on the state management in the voice loop
             pass
        
    
    def wait_drone(self):
        while True:
            try:
                if self.controller.is_connected() and (not self.current_agent_task_thread or not self.current_agent_task_thread.is_alive()):
                    self.controller.drone.api.takeoff()
            except Exception as e:
                print(f"Error during sustained takeoff: {e}")
            time.sleep(2)

    def _voice_listening_loop(self):
        """Main loop for wake word detection and command processing."""
        if not self.whisper_model:
             print("Voice loop cannot start: Whisper model not loaded.")
             self.update_voice_status("Voice Disabled (No Model)")
             return

        print(f"Voice loop started. Listening for wake word: '{WAKE_WORD}'")
        self.is_listening_for_command = False # Start by listening for wake word
        self.wait_thread = threading.Thread(target=self.wait_drone)
        self.wait_thread.start()
        while not self._voice_thread_stop_event.is_set():
            try:
                if self.is_listening_for_command:
                     # --- Listening for Command ---
                     self.update_voice_status("Listening for Command...")
                     command_audio = self._listen_for_audio(AUDIO_DURATION_COMMAND)
                     if command_audio is None or self._voice_thread_stop_event.is_set(): continue # Check stop event after blocking listen

                     command_text = self._transcribe_audio(command_audio)
                     if self._voice_thread_stop_event.is_set(): break # Check stop event

                     if command_text:
                        print(f"Command Heard: '{command_text}'")
                        self.update_last_command(command_text)
                        # Don't speak confirmation here, trigger task will speak
                        self._trigger_agent_task(command_text)
                     else: 
                        print("Did not catch command.")
                        self._speak("I didn't catch that command.")

                     # Reset state after command attempt
                     self.is_listening_for_command = False


                else:
                     # --- Listening for Wake Word ---
                     # Update status less frequently here if needed
                     self.update_voice_status(f"Listening for '{WAKE_WORD}'...")
                     audio_data = self._listen_for_audio(AUDIO_DURATION_LISTEN)
                     if audio_data is None or self._voice_thread_stop_event.is_set(): continue

                     transcribed_text = self._transcribe_audio(audio_data)
                     if self._voice_thread_stop_event.is_set(): break

                     # print(f"Wake Word Check Heard: {transcribed_text}") # Can be noisy

                     if WAKE_WORD in transcribed_text:
                          print("Wake word detected!")
                          self._speak("Ready for command") # Speak confirmation
                          self.is_listening_for_command = True # Switch state
                     else: # Optional: Check for cancel command even when not listening for command
                        if "drone cancel" in transcribed_text or "cancel task" in transcribed_text:
                            if self.current_agent_task_thread and self.current_agent_task_thread.is_alive():
                                print("Cancel command detected (during wake word listen)!")
                                self._speak("Cancelling task")
                                if hasattr(self.controller, 'cancel_task'):
                                    self.controller.cancel_task()

            except Exception as e:
                self.wait_thread.join()
                print(f"Error in voice listening loop: {e}")
                import traceback
                traceback.print_exc()
                self.update_voice_status("Voice Loop Error")
                time.sleep(2) # Pause after error

        print("Voice listening thread finished.")
        self.update_voice_status("Voice Disabled.")


    def _trigger_agent_task(self, command_text):
        """Starts the agent task execution in a background thread."""
        if not self.controller.is_connected():
            self._speak("Cannot execute command, drone is not connected.")
            return

        if self.current_agent_task_thread and self.current_agent_task_thread.is_alive():
             # Handle case where a task is already running. Options:
             # 1. Reject new command:
             self._speak("Sorry, I'm already busy with another task.")
             print(f"Agent task rejected: Already running.")
             return
             # 2. Cancel previous task and start new one (more complex):
             # print("Cancelling previous task to start new one...")
             # self.controller.cancel_task()
             # self.current_agent_task_thread.join(timeout=2.0) # Wait briefly for cancel
             # # Proceed to start new task...

        # Check for simple direct commands first (takeoff/land might not need LLM)
        if command_text.strip().lower() == "takeoff":
             print("Executing direct takeoff...")
             self.update_agent_status("Task: Takeoff")
             self._speak("Taking off.")
             # Define target function for thread
             def run_takeoff():
                  try:
                       # Use parsed command format
                       self.controller.drone.execute_commands([{"command": "takeoff", "params": {}}])
                       self.update_agent_status("Takeoff Complete.")
                  except Exception as e:
                       print(f"Direct takeoff error: {e}")
                       self.update_agent_status("Takeoff Failed!")
                       self._speak("Takeoff failed.")
             # Start thread
             self.current_agent_task_thread = threading.Thread(target=run_takeoff, daemon=True)
             self.current_agent_task_thread.start()

        elif command_text.strip().lower() == "land":
              print("Executing direct land...")
              self.update_agent_status("Task: Landing")
              self._speak("Landing.")
              def run_land():
                  try:
                       self.controller.land_drone_immediately() # Use agent's immediate land
                       self.update_agent_status("Landed.")
                  except Exception as e:
                       print(f"Direct land error: {e}")
                       self.update_agent_status("Land Failed!")
                       self._speak("Landing failed.")
              self.current_agent_task_thread = threading.Thread(target=run_land, daemon=True)
              self.current_agent_task_thread.start()

        else:
             # For other commands, use the full agent task execution
             print(f"Starting agent task for command: '{command_text}'")
             self.update_agent_status(f"Task: {command_text[:20]}...")
             self._speak(f"Command confirm, executing: {command_text}")

             # Define target function for thread
             def run_task_in_thread():
                 task_success = False
                 try:
                      # Call the agent's main execution method
                      task_success, analysis = self.controller.execute_task(command_text, max_steps=15) # Limit steps
                 except Exception as e:
                      print(f"Error during agent task execution: {e}")
                      import traceback
                      traceback.print_exc()
                      self.root.after(0, self.update_agent_status, "Task Error!") # Update GUI thread-safely
                      self.root.after(0, self._speak, "An error occurred during the task.") # Update GUI thread-safely
                 finally:
                      if analysis and 'goal_status' in analysis:
                          self._speak(analysis['goal_status'])
                      # Update status after task finishes (success or fail)
                      final_status = "Task Completed." if task_success else "Task Failed/Ended."
                    #   # Check if cancelled
                    #   if self.controller._cancel_event.is_set():
                    #       final_status = "Task Cancelled."

                      # Schedule GUI update from this thread
                      self.root.after(0, self.update_agent_status, final_status)
                    #   self.root.after(0, self._speak, final_status)


             # Start the agent task thread
             self.current_agent_task_thread = threading.Thread(target=run_task_in_thread, daemon=True)
             self.current_agent_task_thread.start()


    # --- GUI Update Callbacks (Thread-safe updates) ---

    def update_status(self, message):
         """Safely updates the main status label from any thread."""
         if hasattr(self, 'status_var') and self.status_var and self.root.winfo_exists():
             self.root.after(0, self.status_var.set, f"Status: {message}")

    def update_voice_status(self, message):
         """Safely updates the voice status label."""
         if hasattr(self, 'voice_status_var') and self.voice_status_var and self.root.winfo_exists():
              self.root.after(0, self.voice_status_var.set, f"Voice: {message}")

    def update_last_command(self, message):
         """Safely updates the last command label."""
         if hasattr(self, 'last_command_var') and self.last_command_var and self.root.winfo_exists():
              self.root.after(0, self.last_command_var.set, f"Last Cmd: {message}")

    def update_agent_status(self, message):
         """Safely updates the agent status label."""
         if hasattr(self, 'agent_status_var') and self.agent_status_var and self.root.winfo_exists():
              self.root.after(0, self.agent_status_var.set, f"Agent: {message}")

    def _update_state_labels(self):
        """Updates the drone state labels in the GUI (called via root.after)."""
        if not self.controller.is_connected() or not self.root.winfo_exists(): return
        try:
            self.battery_var.set(f"{self.current_bat}%")
            self.height_var.set(f"{self.current_h} cm (AGL)") # Clarify AGL
            self.tof_var.set(f"{self.current_tof} cm")
            self.attitude_var.set(f"P:{self.current_pitch} R:{self.current_roll} Y:{self.current_yaw}") # Compact
            self.speed_var.set(f"X:{self.current_vgx} Y:{self.current_vgy} Z:{self.current_vgz}") # Compact
            self.time_var.set(f"{self.current_time}s")
            # self.odom_var.set(...) # Update cmd odom if still calculating it
        except tk.TclError as e:
             # print(f"Minor GUI update error (state labels): {e}")
             pass # Ignore if closing


    def _update_video_label_image(self, imgtk):
        """Updates the video label with a new image (called via root.after)."""
        if hasattr(self, 'video_label') and self.video_label.winfo_exists():
            self.video_label.config(image=imgtk, text="")
            self.video_label.image = imgtk # Keep reference
        # else: print("Video label DNE") # Debug


    def _update_video_label_text(self, text):
         """Updates the video label with text (called via root.after)."""
         if hasattr(self, 'video_label') and self.video_label.winfo_exists():
              self.video_label.config(text=text, image="")
              self.video_label.image = None


    # --- Window Closing ---
    def on_closing(self):
        """Handles the window close event."""
        print("Close button clicked.")
        # Gracefully disconnect agent system (stops threads, lands drone)
        self.disconnect_agent_system()

        # Optional: Explicitly close voice components if needed
        # Whisper model might not need explicit closing
        # OpenAI client doesn't usually need explicit closing

        # Destroy the main window
        print("Destroying Tkinter root...")
        self.root.destroy()
        print("Application Root Destroyed.")


# --- Main Execution ---
if __name__ == "__main__":
    try:
        print("Starting Tello Vision Agent GUI...")
        root = tk.Tk()
        app = TelloGUIApp(root)
        root.mainloop() # Start Tkinter event loop
    except Exception as main_err:
         print(f"\n--- FATAL GUI ERROR ---")
         print(f"An unexpected error occurred: {main_err}")
         import traceback
         traceback.print_exc()
    finally:
         print("Application Exited.")