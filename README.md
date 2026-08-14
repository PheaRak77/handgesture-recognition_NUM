# Khmer Gesture Recognition

A real-time gesture recognition application that detects hand gestures and displays their Khmer translations.

## Features

- Real-time hand gesture detection using MediaPipe
- Support for multiple cameras
- Khmer text rendering for gesture meanings
- Multi-threaded camera processing
- Demo mode for testing with video files

## CSL phrase gestures (from reference charts)

### One hand
| Gesture | Khmer | How to sign |
|---------|-------|-------------|
| **How Are You** | សុខសប្បាយ | Thumb up at chest |
| **Wrong** | ខុស | Index finger toward cheek |
| **Understand** | យល់ | Index finger at temple |
| **Hearing** | ស្តាប់ឮ | Index finger toward ear |
| **Deaf** | ថ្លង់ | Index toward ear or mouth |
| **Sorry** | សុំទោស | One flat hand on chest, small rub |

### Two hands
| Gesture | Khmer | How to sign |
|---------|-------|-------------|
| **Thank You** | អរគុណ | Left palm up; right hand taps down on left palm |
| **Again** | ម្តងទៀត | Same as thank, then small side-to-side rub on palm |
| **Right** | ត្រូវ | Both index fingers up; right index above left, tap down |
| **Congratulation** | អបអរសាទរ | Both hands up at shoulders, palms forward, fingers spread |
| **Please** | សូម | Both palms up, arms apart |
| **Hello** | សួស្តី | Both hands open, same height, apart |
| **Where From** | មកពីណា | Both point up, or one point + one open palm |

**Tips:** Both hands must be visible for two-hand signs. Hold each pose 1–2 seconds.

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

### PostgreSQL logging
```bash
# 1. Copy env.example to .env and set your pgAdmin database (e.g. db_num_project)
# 2. Seed gestures in pgAdmin: run database/02_seed_gestures.sql
# 3. Test connection
python scripts/test_db.py

# 4. Run with database logging
python main.py --db
```

### Train smarter gesture detection (ML)

The app uses **MediaPipe landmarks + a trained classifier** (with rule-based fallback).
Per-camera settings live in `config/cameras.yaml` (mirror, smoothing, ML threshold).

```bash
source venv/bin/activate
pip install -r requirements.txt

# 1. Record samples (50+ per gesture recommended)
python scripts/collect_landmarks.py --camera 0
#    [ / ] = change gesture label   SPACE = record 30 frames   q = quit

# 2. Train classifier → models/gesture_classifier.joblib
python scripts/train_classifier.py

# 3. Run live (ML loads automatically when model exists)
python main.py --camera 0
python main.py --camera 1   # phone cam — mirror settings in cameras.yaml
```

Edit `config/cameras.yaml` to tune `min_phrase_frames`, `ml_confidence`, and `mirror` per camera.

When a gesture changes, one row is inserted into `gesture_events`. View in pgAdmin:
```sql
SELECT g.name_en, g.text_khmer, e.detected_at
FROM gesture_events e
JOIN gestures g ON g.id = e.gesture_id
ORDER BY e.detected_at DESC;
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
- PostgreSQL (optional, for `--db` logging)
- psycopg2-binary, python-dotenv (optional)
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

```