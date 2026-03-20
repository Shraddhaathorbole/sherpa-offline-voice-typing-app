import tkinter as tk
from app.gui import VoiceTypingGUI


def launch_app():
    root = tk.Tk()
    VoiceTypingGUI(root)
    root.mainloop()