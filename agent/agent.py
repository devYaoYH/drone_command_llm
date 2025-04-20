import os
import sys
import time
import cv2
import requests
from agent.drone_api import Drone
import threading
import base64
import json
import requests

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_odom.config import CONTROL_DISTANCE_CM

class TelloDroneVisionController:
    def __init__(self, server_url="http://localhost:8000", gui_instance=None):
        """Initialize the Tello drone controller with vision server integration"""
        self.server_url = server_url
        self.drone = Drone()
        self.connected = False
        self.gui_instance = gui_instance
        
        # Safety parameters
        self.default_distance = 30  # cm
        self.max_distance = 100     # cm
        self.max_angle = 180        # degrees
        
        # Task tracking
        self.goal_prompt = ""
        self.past_actions = []
        self.planned_actions = []
        self.goal_reached = False
        
        # Initialize image thread as None
        self.img_thread = None
        
        # Map actions to lambda functions for cleaner execution
        self.action_mapping = {
            # Basic commands (no parameters)
            "land": lambda _: ("land", {}),
            "takeoff": lambda _: ("takeoff", {}),
            
            # Movement commands (with distance)
            "up": lambda d: ("move_up", {"distance": min(max(int(d), 20), self.max_distance)}),
            "down": lambda d: ("move_down", {"distance": min(max(int(d), 20), self.max_distance)}),
            "left": lambda d: ("move_left", {"distance": min(max(int(d), 20), self.max_distance)}),
            "right": lambda d: ("move_right", {"distance": min(max(int(d), 20), self.max_distance)}),
            "forward": lambda d: ("move_forward", {"distance": min(max(int(d), 20), self.max_distance)}),
            "back": lambda d: ("move_back", {"distance": min(max(int(d), 20), self.max_distance)}),
            "backward": lambda d: ("move_back", {"distance": min(max(int(d), 20), self.max_distance)}),
            
            # Rotation commands (with angle)
            "cw": lambda a: ("rotate_clockwise", {"angle": min(max(int(a), 1), self.max_angle)}),
            "ccw": lambda a: ("rotate_counter_clockwise", {"angle": min(max(int(a), 1), self.max_angle)})
        }

    def is_connected(self):
        """Return the connection status of the drone"""
        return self.connected

    def connect(self):
        """Connect to the drone and check server connectivity"""
        try:
            # Check server health
            health_response = requests.get(f"{self.server_url}/health")
            if health_response.status_code != 200:
                print(f"Warning: Server health check failed with status {health_response.status_code}")
                return False
            print(f"Server is healthy: {health_response.json()}")
            
            # Connect to drone
            self.connected = self.drone.connect_drone()
            if self.connected:
                self.img_thread = threading.Thread(target=self.save_image)
                self.img_thread.start()
            return self.connected
        except Exception as e:
            print(f"Connection error: {e}")
            return False
            
    def _capture_image(self):
        """Capture image from drone and convert to base64"""
        if not self.connected:
            print("Drone not connected. Please connect first.")
            return None
            
        # Capture image
        frame = self.drone.capture_image()
        if frame is None:
            print("Failed to capture image from drone")
            return None
            
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64

    def capture_and_analyze(self, prompt="take off then turn 1m square then land"):
        """Initial analysis of the task - returns list of planned actions"""
        if not self.connected:
            print("Drone not connected. Please connect first.")
            return None
        
        # Reset task tracking
        self.goal_prompt = prompt
        self.past_actions = []
        self.planned_actions = []
        self.goal_reached = False
            
        # Capture image
        img_base64 = self._capture_image()
        if img_base64 is None:
            return None
        
        # Send to server for analysis
        try:
            response = requests.post(
                f"{self.server_url}/analyze",
                json={
                    "image": img_base64,
                    "prompt": prompt,
                    "structured_output": True,
                    "is_flying": self.drone.is_flying()
                }
            )

            if response.status_code != 200:
                print(f"Error from server: {response.status_code} - {response.text}")
                return None
                
            # Get action plan from server
            result = response.json()
            actions = result.get("result", [])
            
            print(f"Initial plan: {actions}")
            actions = self._parse_actions(actions)
            self.planned_actions = actions

            # Execute first action
            self.drone.execute_commands(actions[:1])
            self.past_actions.append(actions[0])

            
            return self.planned_actions
        except Exception as e:
            print(f"Error sending image to server: {e}")
            return None

    def _parse_actions(self, action_strings):
        """Parse action strings into command objects"""
        commands = []
        
        for action_str in action_strings:
            # Parse action string (e.g., "forward 50" or "cw 90")
            parts = action_str.strip().split()
            if not parts:
                continue
                
            action_type = parts[0].lower()
            
            # Extract value if provided, otherwise use default
            value = self.default_distance if len(parts) <= 1 else parts[1]
            
            # Apply action if valid
            if action_type in self.action_mapping:
                command, params = self.action_mapping[action_type](value)
                # Ensure params is always a dict for API compatibility
                commands.append({"command": command, "params": params})
            else:
                print(f"Unknown action type: {action_type}")
        
        return commands

    def execute_next_step(self):
        """Execute the next step in the plan or get a new one based on current state"""
        if not self.connected:
            print("Drone not connected. Please connect first.")
            return False
            
        if self.goal_reached:
            print("Goal already reached. Start a new task.")
            return False
            
        # Capture current image
        img_base64 = self._capture_image()
        if img_base64 is None:
            return False

        analysis = dict()
            
        # Decide next action
        try:
            # Format past actions for API - make sure they include required params field
            past_actions_api = []
            for action in self.past_actions:
                api_action = f"{action['command']}{action['params']}"
                past_actions_api.append(api_action)
            
            # Format planned actions for API - make sure they include required params field
            planned_actions_api = []
            for action in self.planned_actions:
                api_action = f"{action['command']}{action['params']}"
                planned_actions_api.append(api_action)
            
            print("Sending to server:")
            print(f"Past actions: {past_actions_api}")
            print(f"Planned actions: {planned_actions_api}")
            
            # Request next step from server
            response = requests.post(
                f"{self.server_url}/get_next_step",
                json={
                    "image": img_base64,
                    "goal_prompt": self.goal_prompt,
                    "past_actions": past_actions_api,
                    "future_actions": planned_actions_api,
                    "local_llm": False,
                    "is_flying": self.drone.is_flying()
                }
            )
            
            if response.status_code != 200:
                print(f"Error from server: {response.status_code} - {response.text}")
                # Print the response content for debugging
                try:
                    error_details = response.json()
                    print(f"Error details: {json.dumps(error_details, indent=2)}")
                except:
                    print(f"Raw error response: {response.text}")
                return False
                
            # Process response
            result = response.json()
            next_action = result.get("next_action", {})
            self.goal_reached = result.get("goal_reached", False)
            analysis = result.get("analysis", {})

            if not self.drone.is_flying():
                next_action = "takeoff"
            
            print(f"Next action: {next_action}")
            print(f"Goal reached: {self.goal_reached}")
            print(f"Analysis: {analysis}")
            
            # Execute the next action
            if next_action:
                # Parse action string (e.g., "forward 50" or "cw 90")
                parts = next_action.strip().split()
                if not parts:
                    print("No valid next action received")
                    return False
                action_type = parts[0].lower()
                
                # Extract value if provided, otherwise use default
                value = self.default_distance if len(parts) <= 1 else parts[1]
                
                # Apply action if valid
                if action_type in self.action_mapping:
                    command, params = self.action_mapping[action_type](value)
                    # Ensure params is always a dict for API compatibility
                    action = {"command": command, "params": params}
                    self.drone.execute_commands([action])
                    
                    value_dist = params.get("distance") or params.get("angle") or CONTROL_DISTANCE_CM
                    
                    #-----
                    if self.gui_instance:
                        # Need to map API command back to the type used in update_command_odometry if names differ
                        gui_command_type = command # Assumes names match for simplicity
                    if value_dist is not None: # Only update if there was a value associated
                        # Schedule the update in the GUI thread
                        self.gui_instance.root.after(0, self.gui_instance.update_command_odometry, gui_command_type, value_dist)
                        self.past_actions.append(action)
                    #------
                else:
                    print(f"Unknown action type: {action_type}")
                return True, analysis
            else:
                print("No valid next action received")
                return False, analysis
                
        except Exception as e:
            print(f"Error getting next step: {e}")
            import traceback
            traceback.print_exc()
            return False, analysis

    def execute_task(self, prompt, max_steps=10):
        """Execute a complete task with iterative steps"""
        # Initial analysis
        analysis = dict()
        initial_plan = self.capture_and_analyze(prompt)
        if not initial_plan:
            print("Failed to create initial plan")
            return False, 'Nothing was done'
        
        # Execute steps until goal is reached or max steps
        step_count = 0
        while not self.goal_reached and step_count < max_steps:
            step_count += 1
            print(f"\nExecuting step {step_count}/{max_steps}...")
            
            success, analysis = self.execute_next_step()
            if not success:
                print(f"Failed to execute step {step_count}")
                break
            
            if self.goal_reached:
                print(f"Goal reached in {step_count} steps!")

        if step_count >= max_steps:
            print(f"Max steps ({max_steps}) reached without completing goal")

        return self.goal_reached, analysis
    
    def post_data_to_server(self, endpoint: str, data: dict):
        """Sends JSON data via POST to a specified server endpoint."""
        if not self.connected:
            print("Agent Error: Cannot send data, not connected to server.")
            return False

        url = self.server_url.rstrip('/') + endpoint
        try:
            response = requests.post(url, json=data, timeout=2.0) # Short timeout
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
            # print(f"Successfully posted data to {endpoint}") # Debug
            return True
        except requests.exceptions.RequestException as e:
            print(f"Agent Error: Failed to post data to {url}: {e}")
            # Consider handling specific errors like connection errors differently
            return False
        except Exception as e:
            print(f"Agent Error: Unexpected error posting data to {url}: {e}")
            return False

    def cancel_task(self):
        self.goal_prompt = ""
        self.past_actions = []
        self.planned_actions = []
        self.goal_reached = False
    
    def save_image(self):
        while True:
            img_base64 = self._capture_image()
            response = requests.post(
                    f"{self.server_url}/cache_image",
                    json={
                        "image": img_base64,
                    }
                )
            time.sleep(0.5)
    
    def get_video_reader(self):
        """Get the video frame reader from the drone."""
        try:
            if not self.connected or not self.drone or not self.drone.api:
                return None
            
            # Ensure stream is on
            if not self.drone.api.stream_on:
                print("Turning video stream on...")
                self.drone.api.streamon()
                time.sleep(0.5)  # Give stream time to initialize
            
            # Get frame reader
            frame_read = self.drone.api.get_frame_read()
            if frame_read is None:
                print("Failed to get frame reader")
                return None
                
            return frame_read
            
        except Exception as e:
            print(f"Error getting video reader: {e}")
            return None

    def __del__(self):
        if self.img_thread is not None and self.img_thread.is_alive():
            self.img_thread.join()
