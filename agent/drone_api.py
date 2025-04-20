import time
from djitellopy import Tello

class Drone():
    def __init__(self):
        self.api = Tello()
    
    def connect_drone(self):
        """Connect to the Tello drone"""
        try:
            print("Connecting to Tello drone...")
            self.api.connect()
            battery = self.api.get_battery()
            print(f"Connected to Tello drone. Battery: {battery}%")
            if battery < 20:
                print("WARNING: Battery level is low. Charging recommended before flight.")
            return True
        except Exception as e:
            print(f"Failed to connect to Tello drone: {e}")
            return False

    def capture_image(self):
        """Capture an image from the drone's camera"""
        try:
            if not self.api.stream_on:
                self.api.streamon()
                time.sleep(2)  # Give time for stream to initialize
            
            frame = self.api.get_frame_read().frame
            if frame is None:
                print("Failed to capture frame from drone")
                return None
                
            self.current_image = frame
            return frame
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None

    def is_flying(self):
        return self.api.is_flying
   
    def execute_commands(self, commands):
        """
        Execute a list of drone commands
        
        Args:
            commands: List of command dictionaries
        """
        if not commands:
            print("No valid commands to execute")
            return
            
        try:
            for i, cmd in enumerate(commands):
                command = cmd.get("command")
                params = cmd.get("params", {})
                
                print(f"Executing step {i+1}/{len(commands)}: {command} {params}")
                
                if command == "takeoff":
                    self.api.takeoff()
                    
                elif command == "land":
                    self.api.land()
                    
                elif command == "wait":
                    duration = params.get("duration", 1)
                    print(f"Waiting for {duration} seconds...")
                    time.sleep(duration)
                    
                elif command == "move_forward":
                    self.api.move_forward(params.get("distance"))
                    
                elif command == "move_back":
                    self.api.move_back(params.get("distance"))
                    
                elif command == "move_left":
                    self.api.move_left(params.get("distance"))
                    
                elif command == "move_right":
                    self.api.move_right(params.get("distance"))
                    
                elif command == "move_up":
                    self.api.move_up(params.get("distance"))
                    
                elif command == "move_down":
                    self.api.move_down(params.get("distance"))
                    
                elif command == "rotate_clockwise":
                    self.api.rotate_clockwise(params.get("angle"))
                    
                elif command == "rotate_counter_clockwise":
                    self.api.rotate_counter_clockwise(params.get("angle"))
                
                # Short delay between commands for stability
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Error executing commands: {e}")
            # Emergency landing if something goes wrong
            try:
                self.api.land()
            except:
                pass