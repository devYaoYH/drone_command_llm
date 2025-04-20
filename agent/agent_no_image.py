import json
import time
from langchain_community.chat_models import ChatOpenAI, ChatGooglePalm
from langchain_core.messages import HumanMessage, SystemMessage

class TelloLanguageCommandSystem:
    def __init__(self, api_key, llm_provider="openai", model_name=None):
        """
        Initialize the Tello drone command system that uses language for control
        
        Args:
            api_key: API key for the chosen LLM provider
            llm_provider: Provider to use ('openai' or 'gemini')
            model_name: Optional specific model name (defaults to best available)
        """
        from agent.drone_api import Drone
        
        self.api_key = api_key
        self.llm_provider = llm_provider.lower()
        self.drone = Drone()
        self.safety_bounds = {
            "max_height": 200,  # cm
            "max_distance": 300,  # cm
            "min_height": 30,   # cm
            "max_speed": 50     # cm/s
        }
        
        # Initialize the appropriate LLM based on provider
        if self.llm_provider == "openai":
            self.model_name = model_name or "gpt-4.1-mini"
            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=api_key,
                temperature=0.0  # Keep deterministic for drone commands
            )
        elif self.llm_provider == "gemini":
            self.model_name = model_name or "gemini-pro"
            self.llm = ChatGooglePalm(
                model=self.model_name,
                google_api_key=api_key,
                temperature=0.0  # Keep deterministic for drone commands
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    def connect_drone(self):
        """Connect to the Tello drone and initialize it"""
        return self.drone.connect_drone()
    
    def generate_prompt(self, command):
        """Generate a prompt for the LLM based on the command"""
        return f"""
You are controlling a Tello drone using the DJITelloPy API. The user has given the following command:

"{command}"

Based on this command, generate a structured sequence of exactly 5 drone movement steps that will execute this command safely.

The drone is currently hovering and awaiting instructions. You need to provide movements from the following allowed operations:
- takeoff()
- land()
- move_forward(x) where x is distance in cm (20-500)
- move_back(x) where x is distance in cm (20-500)
- move_left(x) where x is distance in cm (20-500)
- move_right(x) where x is distance in cm (20-500)
- move_up(x) where x is distance in cm (20-500)
- move_down(x) where x is distance in cm (20-500)
- rotate_clockwise(x) where x is angle in degrees (1-360)
- rotate_counter_clockwise(x) where x is angle in degrees (1-360)

Safety constraints:
- Keep movements within 20-200 cm per step
- Use reasonable speeds and distances
- Without visual information, be conservative in your movements
- Make movements more incremental when precision is needed

Respond with ONLY a JSON array of exactly 5 command steps with no additional explanation, formatted as follows:
```json
[
  {{"command": "takeoff", "params": {{}}}},
  {{"command": "move_forward", "params": {{"distance": 50}}}},
  ...
]
```

If a step isn't needed, use a "wait" command with a duration in seconds:
{{"command": "wait", "params": {{"duration": 2}}}}

Focus on accomplishing the goal in the safest, most efficient manner without visual feedback.
"""

    def query_llm(self, prompt):
        """
        Query the language model with a text prompt using LangChain
        
        Args:
            prompt: Text prompt for the LLM
            
        Returns:
            Structured response from the LLM
        """
        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error querying language model: {e}")
            return None
    
    def validate_and_sanitize_commands(self, commands):
        """
        Validate commands for safety and sanitize parameters
        
        Args:
            commands: List of command dictionaries
            
        Returns:
            Sanitized and validated commands
        """
        validated_commands = []
        
        for cmd in commands:
            command = cmd.get("command", "").strip()
            params = cmd.get("params", {})
            
            if command == "takeoff":
                validated_commands.append({"command": "takeoff", "params": {}})
                
            elif command == "land":
                validated_commands.append({"command": "land", "params": {}})
                
            elif command == "wait":
                duration = min(max(params.get("duration", 1), 0.5), 5)  # Cap wait time between 0.5-5 seconds
                validated_commands.append({"command": "wait", "params": {"duration": duration}})
                
            elif command in ["move_forward", "move_back", "move_left", "move_right"]:
                distance = params.get("distance", 0)
                # Ensure distance is within safe bounds
                distance = min(max(int(distance), 20), self.safety_bounds["max_distance"])
                validated_commands.append({"command": command, "params": {"distance": distance}})
                
            elif command in ["move_up", "move_down"]:
                distance = params.get("distance", 0)
                # More restrictive for vertical movements
                distance = min(max(int(distance), 20), 100)
                validated_commands.append({"command": command, "params": {"distance": distance}})
                
            elif command in ["rotate_clockwise", "rotate_counter_clockwise"]:
                angle = params.get("angle", 0)
                angle = min(max(int(angle), 1), 360)
                validated_commands.append({"command": command, "params": {"angle": angle}})
        
        return validated_commands
    
    def process_command(self, user_command):
        """
        Process a user command using language model
        
        Args:
            user_command: Natural language command from user
        """
        # Step 1: Generate prompt and query LLM
        prompt = self.generate_prompt(user_command)
        response = self.query_llm(prompt)
        
        if not response:
            print("Failed to get response from language model")
            return
        
        # Step 2: Parse the JSON response
        try:
            # Extract JSON part if surrounded by code blocks
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].strip()
            else:
                json_text = response.strip()
                
            commands = json.loads(json_text)
            
            # Step 3: Validate and sanitize commands
            validated_commands = self.validate_and_sanitize_commands(commands)
            
            # Step 4: Execute commands
            self.drone.execute_commands(validated_commands)
            
        except Exception as e:
            print(f"Error processing command: {e}")
            print(f"Raw response: {response}")