import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from openai import OpenAI
from agent.agent import TelloDroneVisionController

# ---- SETUP ----

# Init Whisper model
WHISPER_SIZE = "base"
NUM_CORES = os.cpu_count()
WAKE_WORD = "hey drone"

whisper_model = WhisperModel(
    WHISPER_SIZE,
    device="cpu",
    compute_type="int8",
    cpu_threads=NUM_CORES,
    num_workers=NUM_CORES
)

# Init OpenAI TTS client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Init drone controller
controller = TelloDroneVisionController(server_url="http://localhost:8000")

# ---- FUNCTIONS ----

def transcribe_audio(audio_data, sample_rate=16000):
    """Convert numpy audio array to text using Whisper"""
    try:
        # Save audio to temporary file
        temp_file = "temp_recording.wav"
        sf.write(temp_file, audio_data, sample_rate)
        
        # Transcribe
        segments, _ = whisper_model.transcribe(temp_file)
        text = ''.join(segment.text for segment in segments).strip().lower()
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return text
    except Exception as e:
        print(f"[!] Transcription error: {e}")
        return ""

def speak(text):
    """Convert text to speech and play it"""
    try:
        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=text
        ) as response:
            with sd.OutputStream(samplerate=24000, channels=1, dtype='int16') as stream:
                for chunk in response.iter_bytes(chunk_size=1024):
                    audio_array = np.frombuffer(chunk, dtype=np.int16)
                    stream.write(audio_array)
    except Exception as e:
        print(f"[!] TTS error: {e}")

def listen_for_audio(duration=5, sample_rate=16000):
    """Record audio for specified duration and return as numpy array"""
    print(f"Listening for {duration} seconds...")
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # Wait until recording is finished
    return audio_data.flatten()

# ---- MAIN SCRIPT ----

if __name__ == "__main__":
    print("Successfully connected to drone and server")
    speak("Drone connected and ready")
    
    # Main listening loop
    try:
        print(f"Listening for wake word: '{WAKE_WORD}'")
        while True:
            # Listen for potential wake word
            audio_data = listen_for_audio(duration=3)
            transcribed_text = transcribe_audio(audio_data)
            
            print(f"Heard: {transcribed_text}")
            
            # Check if wake word is in the transcribed text
            if WAKE_WORD in transcribed_text:
                print("Wake word detected! Listening for command...")
                speak("Ready for command")
                
                # Listen for the actual command
                command_audio = listen_for_audio(duration=5)
                command_text = transcribe_audio(command_audio)
                
                if command_text:
                    print(f"Command received: {command_text}")
                    speak(f"Executing: {command_text}")
                    
                    # Execute the command in a separate thread
                    import threading
                    
                    # Flag to signal cancellation
                    cancel_requested = threading.Event()
                    
                    # Function to execute task in background
                    def run_task():
                        try:
                            return controller.execute_task(command_text, max_steps=120)
                        except Exception as e:
                            print(f"Task execution error: {e}")
                            return False
                    
                    # Start task execution in background
                    task_thread = threading.Thread(target=run_task)
                    task_thread.daemon = True
                    task_thread.start()
                    
                    # Listen for cancel command while task is running
                    while task_thread.is_alive():
                        # Listen for shorter duration to be responsive
                        cancel_audio = listen_for_audio(duration=1)
                        cancel_text = transcribe_audio(cancel_audio)
                        
                        if "drone cancel" in cancel_text.lower():
                            print("Cancel command detected!")
                            speak("Cancelling task")
                            
                            # Signal controller to stop (you'll need to implement this)
                            controller.cancel_task()
                            
                            # Wait for the task to finish cleaning up
                            task_thread.join(timeout=5.0)
                            break
                    
                    # Check if task completed successfully
                    if task_thread.is_alive():
                        speak("Task could not be cancelled cleanly")
                    else:
                        speak("Task completed or cancelled")
                else:
                    speak("I didn't catch that command")
                    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        # Land the drone if it's flying
        try:
            controller.execute_task("land", max_steps=30)
            speak("Drone landed")
        except Exception as e:
            print(f"Error during landing: {e}")
        print("Program ended")