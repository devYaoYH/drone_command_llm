# main.py
"""Main entry point for the Tello Vision Agent Controller application."""

import tkinter as tk
# --- MODIFICATION: Import from the new file name ---
from gui_voice import TelloGUIApp # Import the main application class
# --- END MODIFICATION ---
import sys
from tkinter import messagebox # Import messagebox for the error dialog

if __name__ == "__main__":
    # Define main_err outside the try block for scope in finally
    main_err = None
    try:
        # Check Python version if needed (optional)
        if sys.version_info < (3, 8): # Raised requirement slightly, common now
             print("Warning: Python 3.8 or newer is recommended for some dependencies.")
             # sys.exit(1) # Optionally exit if version is too old

        print("Starting Tello Vision Agent GUI Application...")
        root = tk.Tk()
        app = TelloGUIApp(root)
        root.mainloop() # Blocks here until the GUI window is closed

    except Exception as e:
         # Catch fatal errors during initialization or main loop
         main_err = e # Store the error
         print(f"\n--- FATAL ERROR ---")
         print(f"An unexpected error occurred: {main_err}")
         import traceback
         traceback.print_exc()
         # Attempt to show an error dialog (might fail if Tkinter itself crashed)
         try:
              # Need to create a temporary minimal root if the main one failed early
              if 'root' not in locals() or not root.winfo_exists():
                  root_err = tk.Tk()
                  root_err.withdraw() # Hide the empty window
                  messagebox.showerror("Fatal Error", f"A critical error occurred:\n{main_err}\n\nSee console for details.", parent=None)
                  root_err.destroy()
              else:
                  messagebox.showerror("Fatal Error", f"A critical error occurred:\n{main_err}\n\nSee console for details.", parent=root)

         except Exception as mb_err:
              print(f"(Could not display error dialog: {mb_err})") # Ignore if showing dialog fails

    finally:
         print("Application Exited.")
         # Optionally exit with non-zero code if an error occurred
         if main_err is not None:
              sys.exit(1)
         else:
              sys.exit(0)