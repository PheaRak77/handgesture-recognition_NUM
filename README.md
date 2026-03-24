# Khmer Gesture Recognition

A real-time gesture recognition application that detects hand gestures and displays their Khmer translations.

## Features

- Real-time hand gesture detection using MediaPipe
- Support for multiple cameras
- Khmer text rendering for gesture meanings
- Multi-threaded camera processing
- Demo mode for testing with video files

## How to Use Two-Hand Gestures

The app now supports **phrase gestures** using both hands simultaneously:

1. **Hello** (សួស្តី): Hold both hands open, palms facing each other like a greeting
2. **How Are You** (សុខសប្បាយជា): One hand open palm up, the other hand pointing up
3. **Where From** (មកពីណា): Both hands pointing, or one pointing and one open
4. **Thank You** (អរគុណ): Bring both open hands together in a prayer position
5. **Please** (សូម): Hold both hands open with palms facing up
6. **Sorry** (សុំទោស): Cross your open hands over your chest

**Tips:**
- Make sure both hands are visible to the camera
- Hold the gesture steady for 1-2 seconds for best recognition
- The app prioritizes two-hand gestures when both hands are detected
- If only one hand is detected, it will show single-hand gestures

## Setup

1. Install Python dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Make sure you have a camera connected.

## Usage

### Live Camera Mode
```bash
python3 main.py
```

### Demo Mode with Video File
```bash
python3 main.py --demo path/to/video.mp4
```

### View Available Gestures
```bash
python3 demo_gestures.py
```

### Help
```bash
python3 main.py --help
```

## Controls

- **Press 'q' in any camera window to quit** (click on the window first to focus it)
- **Or press Ctrl+C in the terminal** to quit gracefully
- **ESC key** also works to quit

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- Pillow
- A camera (webcam) for live mode
- Khmer font (Hanuman.ttf included)

## Troubleshooting

- **Can't close the camera/application**: 
  - Click on the OpenCV window to focus it, then press 'q' or ESC
  - Or press Ctrl+C in the terminal
  - The app will show "shutting down gracefully" messages
- **No camera found**: Connect a webcam or use demo mode with a video file
- **Import errors**: Make sure all dependencies are installed in the virtual environment
- **Font issues**: The app includes a Khmer font, but system fonts will be used as fallback

## Project Structure

```
.
├── main.py                 # Main application
├── requirements.txt        # Python dependencies
├── assets/
│   └── fonts/
│       └── Hanuman.ttf     # Khmer font
├── src/
│   ├── gesture_engine.py   # Gesture detection logic
│   ├── khmer_text.py       # Khmer text rendering
│   └── __init__.py
├── data/                   # Data files (empty)
└── models/                 # Model files (empty)
```