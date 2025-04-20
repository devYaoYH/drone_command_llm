# server.py
# HTTP Server for Vision Analysis with LangChain + Gemini

import os
import json
import base64
import tempfile
import logging
from typing import Dict, Any, Optional
import uuid
import time
import numpy as np
import ollama
from collections import deque
from pydantic import BaseModel, Field, validator
from typing import List, Tuple, Optional, Dict, Any # Ensure List, Tuple, Optional are imported
# FastAPI for server implementation
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel, Field

# For image handling
from PIL import Image
import io
import cv2
import google.generativeai as genai
# import ollama

MODEL_NAME = 'moondream:latest'
MAX_CACHE_SIZE = 50  # Store last 50 images
PROMPT_WINDOW_SIZE = 5  # Only use last 5 images in prompts

# Global state for detections
latest_detections = None

# Image cache for storing recent frames
class ImageCacheEntry:
    def __init__(self, image, timestamp):
        self.image = image  # PIL Image
        self.timestamp = timestamp

image_cache = deque(maxlen=MAX_CACHE_SIZE)

class EnvironmentCacheEntry:
    def __init__(self, analysis, timestamp):
        self.analysis = analysis
        self.timestamp = timestamp

environment_cache = deque(maxlen=MAX_CACHE_SIZE)

# --- Pydantic Models ---

class DetectionBox(BaseModel):
    box: List[float] # Should be [xmin, ymin, xmax, ymax] normalized
    label: str
    score: float

    @validator('box')
    def box_must_have_four_elements(cls, v):
        if len(v) != 4:
            raise ValueError('box must contain 4 float elements [xmin, ymin, xmax, ymax]')
        # Add normalization check if desired
        # if not all(0.0 <= x <= 1.0 for x in v):
        #    raise ValueError('box coordinates must be normalized between 0.0 and 1.0')
        return v

class DetectionsRequest(BaseModel):
    detections: Optional[List[DetectionBox]] = None # Allow empty list or null

def add_to_cache(image: Image.Image):
    """
    Add an image to the cache with current timestamp and save as latest image
    
    Args:
        image: PIL Image to cache
    """
    # Add to memory cache
    image_cache.append(ImageCacheEntry(image, time.time()))
    
    # Save as latest image
    save_latest_image(image)

def get_cached_images(limit: int = PROMPT_WINDOW_SIZE):
    """
    Get cached images with their timestamps, limited to the most recent 'limit' entries.
    Returns images in reverse chronological order (newest first).
    
    Args:
        limit (int): Maximum number of images to return, defaults to PROMPT_WINDOW_SIZE
        
    Returns:
        list: List of (image, timestamp) tuples, most recent first
    """
    # Convert deque to list and get the last 'limit' entries
    cached = list(image_cache)
    if not cached:
        return []
    
    # Get the most recent 'limit' entries and reverse them so newest is first
    recent_entries = cached[-min(limit, len(cached)):]
    recent_entries.reverse()
    
    return [(entry.image, entry.timestamp) for entry in recent_entries]

def get_latest_cached_image():
    """Get the most recent image from cache"""
    return image_cache[-1] if image_cache else None

# --- Configuration ---
# IMPORTANT: Set your API Key
# Option 1: Set Environment Variable GOOGLE_API_KEY
# Option 2: Configure directly (less secure for shared code)
# genai.configure(api_key="YOUR_API_KEY")
try:
    # Attempt to configure from environment variable
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    print("API Key configured successfully from environment variable.")
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set the GOOGLE_API_KEY environment variable or configure the API key directly in the script.")
    # You might want to exit or raise an error here in a real application
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred during API key configuration: {e}")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vision-server")

# Initialize the FastAPI app
app = FastAPI(
    title="Vision Analysis API",
    description="API for analyzing images with LangChain + Gemini",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a directory for temporary image storage
TEMP_DIR = os.path.join(tempfile.gettempdir(), "vision_server")
LATEST_IMAGE_PATH = os.path.join(TEMP_DIR, "latest.jpg")
LATEST_PLOT = os.path.join(TEMP_DIR, "tello_plot.png")

os.makedirs(TEMP_DIR, exist_ok=True)

# Global state
latest_analysis = None
command_history = []  # List of recent commands

# Global variables for rate limiting
last_request_time = 0
MIN_REQUEST_INTERVAL = 1.0  # 1 second between requests


PLOT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Plot Viewer</title>
    <style>
        body {
            background-color: #fafafa;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            font-family: Arial, sans-serif;
        }
        h1 {
            color: #333;
            margin: 20px;
        }
        .plot-container {
            width: 90%;
            max-width: 1000px;
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        #plot {
            width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .refresh-info {
            color: #666;
            margin-top: 10px;
            font-size: 0.9em;
        }
    </style>
    <script>
        function refreshPlot() {
            const img = document.getElementById('plot');
            img.src = '/latest_plot?' + new Date().getTime();
        }
        document.addEventListener('DOMContentLoaded', function () {
            refreshPlot();
            setInterval(refreshPlot, 1000);
        });
    </script>
</head>
<body>
    <h1>Live Plot Viewer</h1>
    <div class="plot-container">
        <img id="plot" src="/latest_plot" alt="Live Plot">
        <div class="refresh-info">This plot refreshes every second.</div>
    </div>
</body>
</html>
"""
# HTML template for the viewer page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Drone Camera View</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            background-color: #f0f0f0;
        }
        .main-container {
            max-width: 1200px;
            width: 100%;
            margin: 0 auto;
        }
        .camera-section {
            width: 100%;
            margin-bottom: 20px;
        }
        .analysis-panel {
            width: 100%;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        h1, h2 {
            color: #333;
            margin-bottom: 20px;
        }
        .image-container {
            background-color: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            position: relative; /* Needed for absolute positioning of overlay */
            width: 100%; /* Ensure container takes width */
            max-width: 960px; /* Or your preferred max width */
            margin-left: auto;
            margin-right: auto;
        }
        img#drone-view { /* Target the image specifically */
            display: block; /* Remove extra space below image */
            width: 100%;
            height: auto;
            border-radius: 4px;
            /* Removed max-height and object-fit for simpler scaling */
        }
        canvas#detection-overlay {
            position: absolute;
            top: 10px; /* Match container padding */
            left: 10px; /* Match container padding */
            width: calc(100% - 20px); /* Adjust width based on padding */
            height: calc(100% - 20px); /* Adjust height based on padding */
            pointer-events: none; /* Allow clicks to pass through */
            border-radius: 4px; /* Match image border */
        }
        img {
            width: 100%;
            height: auto;
            border-radius: 4px;
            max-height: 600px;
            object-fit: contain;
        }
        .refresh-text {
            color: #666;
            margin-top: 10px;
            font-size: 0.9em;
            text-align: center;
        }
        .analysis-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .command-list {
            list-style: none;
            padding: 0;
            max-height: 300px;
            overflow-y: auto;
        }
        .command-item {
            padding: 8px;
            margin: 4px 0;
            background: #f8f8f8;
            border-radius: 4px;
            border-left: 4px solid #4CAF50;
        }
        .goal-reached {
            color: #4CAF50;
            font-weight: bold;
        }
        pre {
            background: #f8f8f8;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            margin: 0;
        }
    </style>
    <script>
        function refreshContent() {
            // Refresh image
            const img = document.getElementById('drone-view');
            img.src = '/latest_image?' + new Date().getTime();
            
            // Refresh analysis and commands
            fetch('/latest_status')
                .then(response => response.json())
                .then(data => {
                    console.log('Latest status:', data); // Debug logging

                    // Update analysis if it exists
                    if (data.analysis) {
                        document.getElementById('goal-status').textContent = data.analysis.goal_status || 'N/A';
                        document.getElementById('current-situation').textContent = data.analysis.current_situation || 'N/A';
                        document.getElementById('safety-assessment').textContent = data.analysis.safety_assessment || 'N/A';
                        document.getElementById('goal-reached').textContent = data.goal_reached ? 'Yes' : 'No';
                        document.getElementById('future-actions').textContent = data.analysis.future_actions || 'N/A';
                    } else {
                        console.log('No analysis data available');
                    }
                    
                    // Update command history if it exists
                    if (data.commands && Array.isArray(data.commands)) {
                        const commandList = document.getElementById('command-list');
                        commandList.innerHTML = '';
                        data.commands.forEach(cmd => {
                            const li = document.createElement('li');
                            li.className = 'command-item';
                            const cmdText = cmd.command + (Object.keys(cmd.params).length > 0 ? 
                                          ' ' + JSON.stringify(cmd.params) : '');
                            li.textContent = `[${cmd.timestamp}] ${cmdText}`;
                            commandList.appendChild(li);
                        });
                        if (data.commands.length === 0) {
                            const li = document.createElement('li');
                            li.className = 'command-item';
                            const cmdText = 'up 50';
                            li.textContent = `[March 15, 2023 at 10:30 AM] ${cmdText}`;
                            commandList.appendChild(li);
                        }
                    } else {
                        console.log('No command history available');
                        document.getElementById('command-list').innerHTML = '<li>No commands executed yet</li>';
                    }
                    // --- Draw Detections ---
                    const canvas = document.getElementById('detection-overlay');
                    const ctx = canvas.getContext('2d');
                    setTimeout(() => {
                        if (img.complete && img.naturalWidth > 0) { // Check if image is loaded
                           // Get the actual displayed size
                           const displayWidth = img.clientWidth;
                           const displayHeight = img.clientHeight;

                            // Adjust canvas CSS dimensions AND drawing dimensions
                            canvas.style.width = displayWidth + 'px';
                            canvas.style.height = displayHeight + 'px';
                            canvas.width = displayWidth; // Set drawing buffer size
                            canvas.height = displayHeight;

                            // Clear previous drawings
                            ctx.clearRect(0, 0, canvas.width, canvas.height);

                            if (data.detections && Array.isArray(data.detections) && data.detections.length > 0) {
                                console.log(`Drawing ${data.detections.length} detections.`);
                                data.detections.forEach(det => {
                                    if (det.box && det.box.length === 4) {
                                        const [xmin, ymin, xmax, ymax] = det.box; // Normalized coords

                                        // Scale normalized coordinates to canvas dimensions
                                        const drawX = xmin * canvas.width;
                                        const drawY = ymin * canvas.height;
                                        const drawWidth = (xmax - xmin) * canvas.width;
                                        const drawHeight = (ymax - ymin) * canvas.height;

                                        // Draw bounding box
                                        ctx.strokeStyle = 'lime'; // Bright green
                                        ctx.lineWidth = 2;
                                        ctx.strokeRect(drawX, drawY, drawWidth, drawHeight);

                                        // Draw label and score
                                        const label = `${det.label} (${det.score.toFixed(2)})`;
                                        ctx.fillStyle = 'lime';
                                        ctx.font = '14px Arial';
                                        // Position text slightly above the box
                                        let textX = drawX;
                                        let textY = drawY - 5;
                                        // Prevent text going off-screen top
                                        if (textY < 10) {
                                            textY = drawY + 15;
                                        }
                                        ctx.fillText(label, textX, textY);
                                    } else {
                                        console.warn("Skipping invalid detection box:", det.box);
                                    }
                                });
                            } else {
                                // console.log("No detections to draw."); // Optional log
                            }
                        } else {
                             console.log("Image not yet loaded for dimension calculation.");
                             // Clear canvas if image isn't ready
                             ctx.clearRect(0, 0, canvas.width, canvas.height);
                        }
                    }, 50); // Small delay (50ms) to allow image dimensions to stabilize
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                    // Clear canvas on error
                    const canvas = document.getElementById('detection-overlay');
                    if (canvas) {
                       const ctx = canvas.getContext('2d');
                       ctx.clearRect(0, 0, canvas.width, canvas.height);
                    }
                });
        }
        
        // Wait for DOM to be fully loaded before starting refresh cycle
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM loaded, starting refresh cycle');
            // Initial refresh
            refreshContent();
            // Start periodic refresh
            setInterval(refreshContent, 500);
        });
    </script>
</head>
<body>
    <div class="main-container">
        <h1>Drone Camera View</h1>
        
        <div class="camera-section">
             <div class="image-container">
                <img id="drone-view" src="/latest_image" alt="Drone Camera Feed">
                <canvas id="detection-overlay"></canvas> <!-- Add the canvas overlay -->
            </div>
        </div>
        
        <div class="analysis-panel">
            <div class="analysis-section">
                <h2>Current Analysis</h2>
                <p><strong>Goal Status:</strong> <span id="goal-status">The yellow box is not visible in the current frame. The drone needs to gain altitude to get a better view of the surroundings.</span></p>
                <p><strong>Current Situation:</strong> <span id="current-situation">The drone is currently at a low altitude, with a blue surface visible in front. There are objects in the background, but the yellow box is not visible.</span></p>
                <p><strong>Safety Assessment:</strong> <span id="safety-assessment">The drone is close to a surface, so it needs to move up to avoid collision.</span></p>
                <p><strong>Goal Reached:</strong> <span id="goal-reached">No</span></p>
            </div>
            <div class="analysis-section">
                <p><strong>Future Actions:</strong> <span id="future-actions">takeoff</span></p>
                <h2>Command History</h2>
                <ul id="command-list" class="command-list">
                    <li class="command-item">Waiting for commands...</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
"""

def save_latest_image(pil_image: Image.Image):
    """Save the latest image to a consistent location"""
    try:
        pil_image.save(LATEST_IMAGE_PATH, "JPEG")
    except Exception as e:
        logger.error(f"Error saving latest image: {e}")

@app.get("/", response_class=HTMLResponse)
async def get_viewer_page():
    """Serve the HTML page that displays the latest image"""
    return HTMLResponse(content=HTML_TEMPLATE)

@app.get("/get_plot", response_class=HTMLResponse)
async def get_viewer_page():
    """Serve the HTML page that displays the latest image"""
    return HTMLResponse(content=PLOT_HTML)

@app.get("/latest_image")
async def get_latest_image():
    """Serve the latest cached image"""
    if os.path.exists(LATEST_IMAGE_PATH):
        return FileResponse(LATEST_IMAGE_PATH, media_type="image/jpeg")
    elif os.path.exists("temp.jpg"):
        return FileResponse("temp.jpg", media_type="image/jpeg")
    else:
        raise HTTPException(status_code=502, detail="No image available")


@app.get("/latest_plot")
async def get_latest_image():
    """Serve the latest cached image"""
    if os.path.exists(LATEST_PLOT):
        return FileResponse(LATEST_PLOT, media_type="image/png")
    elif os.path.exists("tello_plot.png"):
        return FileResponse("tello_plot.png", media_type="image/png")
    else:
        raise HTTPException(status_code=502, detail="No image available")

# Pydantic models for request and response validation
class ImageRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image data from OpenCV")
    prompt: str = Field("Analyze this image", description="Text prompt for image analysis")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt for the vision model")
    structured_output: bool = Field(True, description="Whether to return structured JSON output")
    local_llm: bool = Field(False, description="Whether to use local LLM")
    is_flying: bool = Field(False, description="Whether the drone is currently flying")

class AnalysisResponse(BaseModel):
    request_id: str
    processing_time: float
    result: Any
    timestamp: str

class NextStepRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image data from OpenCV")
    goal_prompt: str = Field(..., description="Original goal/objective for the drone")
    past_actions: list[str] = Field(default_factory=list, description="List of actions already taken")
    future_actions: list[str] = Field(default_factory=list, description="Planned future actions")
    local_llm: bool = Field(False, description="Whether to use local LLM")
    is_flying: bool = Field(False, description="Whether the drone is currently flying")

class NextStepResponse(BaseModel):
    request_id: str
    processing_time: float
    next_action: str
    goal_reached: bool
    analysis: Dict[str, Any]
    timestamp: str

class CacheImageRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image data from OpenCV")

class CacheImageResponse(BaseModel):
    request_id: str
    timestamp: str
    cache_size: int
    processing_time: float

def get_vision_model(system_prompt=None, structured_output=False):
    """
    Initializes and returns a Gemini Vision model instance.

    Args:
        system_prompt (str, optional): Instructions guiding the model's behavior and output format.
        structured_output (bool): This flag influences how the system_prompt should be crafted
                                  (e.g., explicitly asking for JSON if True).

    Returns:
        genai.GenerativeModel: The initialized model instance, or None on failure.
    """
    # Use a recent model capable of vision input
    model_name = "gemini-2.0-flash"
    # model_name = "gemini-pro-vision" # Older alternative

    # Generation config (optional - control creativity, length etc.)
    generation_config = genai.GenerationConfig(
        temperature=0.2, # Lower temperature for more deterministic/predictable actions
        max_output_tokens=500, # Limit the length of the action sequence
        response_mime_type="application/json" # Structured output
    )

    print(f"Initializing model: {model_name}")
    try:
        model = genai.GenerativeModel(
            model_name,
            system_instruction=system_prompt, # Use the system_instruction parameter
            generation_config=generation_config
        )
        print("Model initialized successfully.")
        return model
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        return None

def prepare_vision_content(image_path, prompt):
    """
    Prepares the content list (image + prompt) for the Gemini API.
    Includes the current image and up to PROMPT_WINDOW_SIZE most recent cached images.
    
    Args:
        image_path (str): Path to the image file.
        prompt (str): The text query for the model.
        
    Returns:
        list: A list containing the text prompt and images with context
    """
    try:
        # Load current image
        current_image = Image.open(image_path)
        
        # Get cached entries (convert deque to list for slicing)
        cached_entries = list(environment_cache)[-5:]  # Get last 5 entries
        cached_entries.reverse()  # Newest first

        # Create the content with temporal context
        time_now = time.time()
        content = [
            f"""Current frame for analysis:
            Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}

            Previous environment analysis for context:
            {[f"{entry.analysis} (at {entry.timestamp})" for entry in cached_entries]}

            User prompt: {prompt}""",
            current_image    # Current image
        ]

        return content
        
    except Exception as e:
        print(f"Error preparing vision content: {e}")
        return None

def parse_action_sequence(response_text):
    """
    Parses the model's text response to extract the action sequence.
    Attempts to parse as JSON list first, falls back to basic checks if needed.

    Args:
        response_text (str): The raw text output from the Gemini model.

    Returns:
        list: A list of validated action strings (e.g., ['forward', 'left']),
              or an empty list if parsing fails or no valid actions are found.
    """
    print(f"Attempting to parse response: '{response_text}'")
    actions = []
    allowed_actions = {"takeoff", "land", "up", "down", "left", "right", "forward", "backward", "ccw", "cw"}

    try:
        # Attempt to parse as JSON (most reliable if prompt requested it)
        potential_actions = json.loads(response_text)["actions"]
        if isinstance(potential_actions, list):
            # Validate that it's a list of strings and actions are allowed
            actions = [str(a).lower() for a in potential_actions if isinstance(a, str) and str(a).lower().split()[0] in allowed_actions]
        else:
            print("Warning: Response parsed as JSON but was not a list.")

    except json.JSONDecodeError:
        print("Warning: Response was not valid JSON. Falling back to basic parsing (less reliable).")
        # Fallback: Clean up potential markdown/quotes and split (less robust)
        cleaned_text = response_text.strip().strip('`').strip('json') # Remove common markdown/prefixes
        cleaned_text = cleaned_text.strip('[]').strip() # Remove potential list brackets
        potential_actions = [a.strip().strip("'\"").lower() for a in cleaned_text.split(',')]
        actions = [a for a in potential_actions if a in allowed_actions]

    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}")

    if not actions:
         print("Warning: Could not extract any valid actions from the response.")

    return actions


# Set up Model endpoint
def query_model(prompt, image_path, system_prompt=None):
    try:
        # Use ollama.chat for a chat-based interaction
        messages = [
            {
                "role": "system",
                "content": system_prompt or DEFAULT_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt,
                "images": [image_path]  # Pass the image path in the message
            }
        ]
        
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            stream=False
        )

        # --- Process the Response ---
        print("\n--- Ollama Response ---")
        if response and isinstance(response, dict) and 'message' in response:
            print(response['message']['content'])
            return response['message']['content']  # Return just the content of the message
        else:
            print("Received an empty or unexpected response format.")
            print(f"Full response object: {response}")
            return None
            
    # --- Handle Potential Errors ---
    except Exception as e:
        print(f"\nAn error occurred while interacting with Ollama: {e}")
        print("Possible causes:")
        print("- Ollama server is not running or stopped.")
        print(f"- Model '{MODEL_NAME}' is not installed or misspelled (`ollama pull {MODEL_NAME}`)")
        print("- Network issue connecting to localhost:11434 (if Ollama runs there).")
        print("- The image file might be corrupted or in an unsupported format.")
    
    return None

def opencv_to_pil(cv_image):
    """Convert OpenCV image format (numpy array) to PIL Image."""
    # OpenCV uses BGR, PIL uses RGB
    color_converted = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(color_converted)

def decode_base64_to_cv_image(base64_string):
    """Decode base64 string to OpenCV image format."""
    # Decode base64 string
    img_data = base64.b64decode(base64_string)
    
    # Convert to numpy array
    nparr = np.frombuffer(img_data, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def save_temp_image(image):
    """Save a PIL image to a temporary file and return the path."""
    # Generate a unique filename
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(TEMP_DIR, filename)
    
    # Save the image
    image.save(filepath)
    image.save("temp.jpg")
    
    return filepath

# Global variables for rate limiting
last_request_time = 0
MIN_REQUEST_INTERVAL = 1.0  # 1 second between requests

# Default system prompt if none provided
DEFAULT_SYSTEM_PROMPT = """
You are an AI assistant specialized in analyzing images to control a flying drone.

Your task is to analyze the image and provide commands to help the drone navigate safely.
You have access to the current frame and the last 5 frames for temporal context.

Always start a sequence of commands with "takeoff" as the first step.

Return your response as a JSON object with the following structure:
{
    "description": "Brief description of what you see and any potential hazards",
    "analysis": {
        "movement": "Describe any movement or changes detected between frames",
        "obstacles": "List any detected obstacles and their positions",
        "safety_concerns": "Note any safety concerns that affect navigation"
    },
    "actions": [
        "command1",
        "command2",
        ...
    ]
}

Available commands and their parameters:
1. Basic Movement:
   - "land": land the drone
   - "takeoff": take off and hover
   - "forward x": move forward x cm
   - "back x": move back x cm
   - "left x": move left x cm
   - "right x": move right x cm
   - "up x": move up x cm
   - "down x": move down x cm

2. Rotation:
   - "cw x": rotate clockwise x degrees
   - "ccw x": rotate counter-clockwise x degrees

Safety Guidelines:
- All distances should be between 20-50 cm
- Vertical movements (up/down) are limited to 20-50 cm
- Rotation angles should be between 1-360 degrees
- Always maintain safe distance from obstacles
- Prefer gradual movements over large sudden changes
- You must first takeoff if the drone is not yet flying
- If you see a wall or a uniform view, move backwards and look around
- Try to do it safely in as little moves as possible
- If you believe you are done, you can ask me a question on what to do next in goal status
- Do not land when the goal is reached, unless told to do so
"""

NEXT_STEP_SYSTEM_PROMPT = """
You are an AI assistant helping a drone navigate towards a goal. Your task is to analyze the current situation and decide the immediate next action.

Context:
1. You will receive the current frame and recent past frames
2. You have the original goal/objective
3. You know what actions have already been taken
4. You have a list of planned future actions

Your task is to:
1. Analyze if the goal has been reached based on the current frame
2. If not reached, determine the best immediate next action considering:
   - Progress towards the goal
   - Safety and obstacles
   - Past actions taken
   - Planned future actions

Return your response as a JSON object with this structure:
{
    "analysis": {
        "goal_status": "Description of progress towards goal",
        "current_situation": "Analysis of current frame",
        "movement_history": "Analysis of past actions",
        "safety_assessment": "Any safety concerns"
    },
    "goal_reached": false,
    "next_action": "command"
}

If you determine the goal has been reached, set goal_reached to true.

Available commands and their parameters:
1. Basic Movement:
   - "land": land the drone
   - "takeoff": take off and hover
   - "forward x": move forward x cm
   - "back x": move back x cm
   - "left x": move left x cm
   - "right x": move right x cm
   - "up x": move up x cm
   - "down x": move down x cm

2. Rotation:
   - "cw x": rotate clockwise x degrees
   - "ccw x": rotate counter-clockwise x degrees

Safety Guidelines:
- All distances should be between 20-50 cm
- Vertical movements (up/down) are limited to 20-50 cm
- Rotation angles should be between 1-360 degrees
- Always maintain safe distance from obstacles
- Prefer gradual movements over large sudden changes
- You must first takeoff if the drone is not yet flying
- If you see a wall or a uniform view, move backwards and look around
- Try to do it safely in as little moves as possible
- If you believe you are done, you can ask me a question on what to do next in goal status
- Do not land when the goal is reached, unless told to do so
"""

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(request: ImageRequest = Body(...)):
    """
    Endpoint to analyze an image frame from OpenCV.
    Expects a base64-encoded image and returns the analysis result.
    """
    global last_request_time
    current_time = time.time()
    
    # Rate limiting check
    if current_time - last_request_time < MIN_REQUEST_INTERVAL:
        raise HTTPException(
            status_code=429, 
            detail="Too many requests. Please wait at least 1 second between requests."
        )
    
    last_request_time = current_time
    start_time = current_time
    request_id = str(uuid.uuid4())
    
    try:
        # Get the system prompt (use default if none provided)
        system_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT
        
        # Decode the base64 image to OpenCV format
        cv_image = decode_base64_to_cv_image(request.image)
        if cv_image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Convert to PIL format
        pil_image = opencv_to_pil(cv_image)
        
        # Add to image cache
        add_to_cache(pil_image)
        
        # Save to a temporary file
        temp_image_path = save_temp_image(pil_image)

        print("Saved file to:", temp_image_path)
        
        # Add drone state to prompt
        drone_state = "The drone is currently " + ("flying" if request.is_flying else "not flying") + "."
        enhanced_prompt = f"{drone_state}\n\n{request.prompt}"

        # Run the analysis
        result = json.loads('[]')
        print(request.local_llm)
        print(request)
        try:
            if request.local_llm:
                result = query_model(enhanced_prompt, temp_image_path, system_prompt=system_prompt)
            else:
                model = get_vision_model(system_prompt=system_prompt)
                content = prepare_vision_content(temp_image_path, enhanced_prompt)
                result = model.generate_content(content).text
        except Exception as e:
            print("Error: ", e)
        print(result)

        actions = ["takeoff"] + parse_action_sequence(result)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Clean up the temporary file
        try:
            os.remove(temp_image_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file: {e}")
        
        # Return the response
        return AnalysisResponse(
            request_id=request_id,
            processing_time=processing_time,
            result=actions,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/update_detections")
async def update_detections(request: DetectionsRequest):
    """Receives detection data from the GUI/Agent and updates global state."""
    global latest_detections
    # print(f"Received detections: {request.detections}") # Debug
    latest_detections = request.detections # Store the list of DetectionBox objects or None
    return {"message": "Detections updated successfully", "count": len(latest_detections) if latest_detections else 0}


@app.post("/get_next_step", response_model=NextStepResponse)
async def get_next_step(request: NextStepRequest = Body(...)):
    """
    Analyze current frame and context to determine the next best action for the drone.
    Takes into account the goal, past actions, and planned future actions.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Decode and process the image
        cv_image = decode_base64_to_cv_image(request.image)
        if cv_image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Convert to PIL and cache
        pil_image = opencv_to_pil(cv_image)
        add_to_cache(pil_image)
        
        # Save to temp file
        temp_image_path = save_temp_image(pil_image)
        
        # Prepare context-rich prompt with drone state
        drone_state = "The drone is currently " + ("flying" if request.is_flying else "not flying") + "."
        context_prompt = f"""
        {drone_state}

        Goal: {request.goal_prompt}

        Past Actions Taken:
        {json.dumps([action for action in request.past_actions], indent=2)}

        Planned Future Actions:
        {json.dumps([action for action in request.future_actions], indent=2)}

        Analyze the current situation and determine the next best action.
        """
        
        # Get model response
        try:
            if request.local_llm:
                result = query_model(context_prompt, temp_image_path, system_prompt=NEXT_STEP_SYSTEM_PROMPT)
            else:
                model = get_vision_model(system_prompt=NEXT_STEP_SYSTEM_PROMPT)
                content = prepare_vision_content(temp_image_path, context_prompt)
                result = model.generate_content(content).text
            
            # Parse the response
            response_data = json.loads(result)
            print(response_data)
            response_data["analysis"]["future_actions"] = request.future_actions
            update_status(response_data["analysis"], response_data["next_action"])
            # Clean up temp file
            try:
                os.remove(temp_image_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")
            
            # Return the response
            return NextStepResponse(
                request_id=request_id,
                processing_time=time.time() - start_time,
                next_action=response_data["next_action"],
                goal_reached=response_data["goal_reached"],
                analysis=response_data["analysis"],
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
        except Exception as e:
            logger.error(f"Error processing next step: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
            
    except Exception as e:
        logger.error(f"Error in get_next_step: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache_image", response_model=CacheImageResponse)
async def cache_image(request: CacheImageRequest = Body(...)):
    """
    Cache a single image frame without performing any analysis.
    Useful for building up context before making analysis requests.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Decode the image
        cv_image = decode_base64_to_cv_image(request.image)
        if cv_image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Convert to PIL and cache
        pil_image = opencv_to_pil(cv_image)
        add_to_cache(pil_image)
        
        return CacheImageResponse(
            request_id=request_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            cache_size=len(image_cache),
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Error caching image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint to verify the server is running."""
    return {"status": "healthy", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

@app.get("/latest_status")
async def get_latest_status():
    """Return the latest analysis, command history, AND detections"""
    global latest_analysis, command_history, latest_detections
    print("[GET LATEST_STATUS]", "latest_analysis:", latest_analysis, "command_history:", command_history, "latest_detections:", latest_detections) # Log detections too
    # Convert DetectionBox objects to dicts for JSON serialization if needed
    detections_serializable = None
    if latest_detections:
        try:
            # Use Pydantic's .dict() or .model_dump() (v2) if they are model instances
            # If they are already dicts from the request processing, this isn't needed.
            # Assuming latest_detections holds list of DetectionBox model instances:
            detections_serializable = [det.model_dump() for det in latest_detections]
            # If it holds dicts already, just assign:
            # detections_serializable = latest_detections
        except AttributeError: # Handle case where they might be dicts already
             detections_serializable = latest_detections
        except Exception as e:
            logger.error(f"Error serializing detections: {e}")
            detections_serializable = [] # Send empty on error


    return {
        "analysis": latest_analysis,
        "goal_reached": latest_analysis.get("goal_reached", False) if latest_analysis else False,
        "commands": command_history,
        "detections": detections_serializable # Add detections here
    }

def update_status(analysis: Dict[str, Any], action: str):
    """Update the latest analysis and command history"""
    global latest_analysis, command_history, environment_cache
    latest_analysis = analysis
    environment_cache.append(EnvironmentCacheEntry(analysis, time.time()))

    # Add command to history with timestamp
    command_entry = {
        "command": action,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {}  # Add parameters if needed
    }
    command_history.append(command_entry)

    # Keep only the last 10 commands
    if len(command_history) > 10:
        command_history.pop(0)

    print("[UPDATE_STATUS]", "latest_analysis:", latest_analysis, "command_history:", command_history)

# Run the server
if __name__ == "__main__":
    # You can adjust host and port as needed
    port = int(os.environ.get("PORT", 8000))
    
    # Log startup message
    logger.info(f"Starting Vision Analysis Server on port {port}")
    logger.info(f"Temporary image directory: {TEMP_DIR}")
    
    # Start the server
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
