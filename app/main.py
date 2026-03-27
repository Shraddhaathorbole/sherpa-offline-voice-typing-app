import logging
import sys
import tkinter as tk
import warnings

# Suppress underlying library deprecation warnings from terminal output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
# Force PyTorch subprocesses (like Mac's POSIX resource_tracker) to inherit silence
os.environ["PYTHONWARNINGS"] = "ignore"

from app.gui import VoiceTypingGUI

def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)

def launch_app() -> None:
    _configure_logging()
    import customtkinter as ctk
    root = ctk.CTk()
    try:
        VoiceTypingGUI(root)
    except Exception:
        logging.exception("Fatal error during application initialisation.")
        sys.exit(1)
    root.mainloop()

if __name__ == "__main__":
    launch_app()
