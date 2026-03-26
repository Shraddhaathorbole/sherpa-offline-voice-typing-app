#!/bin/bash
# Final Production Build Script (Mac OS)
# This will physically convert your Python project into a double-clickable "Offline Voice Typing.app" 
# native to your Macintosh, so you never have to launch from the terminal again!

echo "Starting Production Build..."

# 1. Install standard packaging tools natively into your local Python environment
pip install pyinstaller

# 2. Package the massive dependencies securely (PyTorch + Torchaudio take the most space)
# We disable terminal consoles (--noconsole) so the app opens purely as a graphical UI.
# We physically bundle the `models` folder where your C++ Sherpa ONNX engine sits.

pyinstaller --noconfirm --log-level=WARN \
    --name "VoiceTyping" \
    --windowed \
    --noconsole \
    --add-data "models:models" \
    --hidden-import "torch" \
    --hidden-import "torchaudio" \
    --hidden-import "sounddevice" \
    --hidden-import "reportlab" \
    --hidden-import "symspellpy" \
    --hidden-import "whisper" \
    --hidden-import "tkinter" \
    run.py

echo "Build Complete!"
echo "You can now find your fully production-ready application inside the new 'dist/' folder!"
echo "Double-click 'VoiceTyping.app' to use it!"
