# main_button.py
"""Main entry point for the Tello Advanced Controller application."""

import tkinter as tk
from gui import TelloGUIApp # Import the main application class
import sys

if __name__ == "__main__":
    try:
        # # Check Python version if needed (Tello might require specific versions)
        # if sys.version_info < (3, 6):
        #     print("Error: Python 3.6 or newer is recommended for this application.")
        #     sys.exit(1)

        print("Starting Tello Controller Application...")
        root = tk.Tk()
        app = TelloGUIApp(root)
        root.mainloop()

    except Exception as main_err:
         # Catch fatal errors during initialization or main loop
         print(f"\n--- FATAL ERROR ---")
         print(f"An unexpected error occurred: {main_err}")
         import traceback
         traceback.print_exc()
         # Optionally show an error dialog before exiting
         try:
              tk.messagebox.showerror("Fatal Error", f"A critical error occurred:\n{main_err}\n\nSee console for details.")
         except Exception:
              pass # Ignore if Tkinter itself failed
    finally:
         print("Application Exited.")
         # Ensure sys.exit is called if needed, especially after errors
         # sys.exit(1 if 'main_err' in locals() else 0)