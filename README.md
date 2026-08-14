# Khmer Sign Language & Hand Gesture Recognition

A real-time hand gesture recognition application powered by **MediaPipe**, **OpenCV**, and **Scikit-Learn**. It detects single-hand and dual-hand gestures and displays their corresponding Khmer translations with high-performance multi-threaded camera processing.

---

## 🌟 Key Features

- **Real-Time Detection**: Multi-threaded pipeline decoupling video frame capture from MediaPipe inference for maximum FPS and zero UI lag.
- **Khmer Sign Language (CSL) Support**: Recognizes both single-hand and dual-hand phrase gestures with custom Khmer unicode font rendering (`Hanuman.ttf`).
- **Machine Learning & Rule-Based Recognition**: Supports landmark feature extraction with a trained `RandomForest` classifier (`models/gesture_classifier.joblib`) and automatic rule-based fallbacks.
- **Multi-Camera & DroidCam Support**: Configurable per-camera parameters (mirroring, confidence thresholds, frame smoothing) in `config/cameras.yaml` and automatic camera discovery via `scripts/find_cameras.py`.
- **Demo Mode**: Play pre-recorded video files to test recognition offline without a physical webcam.
- **Optional PostgreSQL Event Logging**: Store gesture detection sessions and real-time detection events directly into a database.

---

## ✋ Supported Gestures

### 🖐️ One-Hand Gestures

| Gesture | Khmer Translation | Sign Description / How to Sign |
| :--- | :--- | :--- |
| **How Are You** | សុខសប្បាយ | Thumb pointing up at chest level |
| **Wrong** | ខុស | Index finger pointing toward cheek |
| **Understand** | យល់ | Index finger touching temple |
| **Hearing** | ស្តាប់ឮ | Index finger pointing toward ear |
| **Deaf** | ថ្លង់ | Index finger pointing toward ear or mouth |
| **Sorry** | សុំទោស | One flat hand placed on chest, moving in small circular rub |

### 👐 Two-Hand Gestures

| Gesture | Khmer Translation | Sign Description / How to Sign |
| :--- | :--- | :--- |
| **Thank You** | អរគុណ | Left palm face up; right hand taps down on left palm |
| **Again** | ម្តងទៀត | Left palm face up; right hand taps and rubs side-to-side on left palm |
| **Right** | ត្រូវ | Both index fingers pointing up; right index above left, tapping down |
| **Congratulation** | អបអរសាទរ | Both hands up near shoulders, palms forward, fingers spread |
| **Please** | សូម | Both palms facing up, arms slightly apart |
| **Hello** | សួស្តី | Both hands open at shoulder height, facing forward |
| **Where From** | មកពីណា | Both index fingers point up, or one index finger + one open palm |

> 💡 **Tip:** Ensure both hands are fully visible in the camera frame for two-hand gestures. Hold each pose steadily for 1–2 seconds for optimal recognition.

---

## 📁 Project Structure

```
handgesture-recognition/
├── assets/
│   └── fonts/
│       └── Hanuman.ttf          # Khmer font for UI rendering
├── config/
│   └── cameras.yaml             # Per-camera configuration (mirror, FPS, threshold)
├── database/
│   └── 02_seed_gestures.sql     # PostgreSQL database schema & seed data
├── models/
│   └── gesture_classifier.joblib# (Generated) Trained Machine Learning model
├── data/
│   └── landmarks/               # (Generated) Collected training CSV datasets
├── scripts/
│   ├── find_cameras.py          # Utility to scan connected webcams / DroidCam IP
│   ├── collect_landmarks.py     # Interactive dataset collection tool
│   ├── train_classifier.py      # ML classifier training script
│   ├── sync_gestures.py         # Sync local gesture mapping with database
│   └── test_db.py               # Database connectivity test utility
├── src/
│   ├── camera_config.py         # YAML camera configuration parser
│   ├── db.py                    # PostgreSQL database handler class
│   ├── gesture_classifier.py    # ML model loader & predictor wrapper
│   ├── gesture_engine.py       # Core MediaPipe landmark recognition logic
│   ├── khmer_text.py            # PIL Khmer font rendering pipeline
│   └── landmark_features.py     # Feature extraction & normalization algorithms
├── demo_gestures.py             # Visual gesture reference catalog GUI
├── main.py                      # Main application entry point
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation
```

---

## 🛠️ Step-by-Step Setup Guide

### Step 1: System Requirements

Ensure your system has the following installed:
- **Python**: Version 3.9 or higher (Python 3.10 / 3.11 recommended)
- **Webcam**: Laptop built-in camera, external USB webcam, or mobile camera via DroidCam
- **PostgreSQL**: (Optional) Required only if database event logging (`--db`) is enabled

### Step 2: Clone & Navigate to Repository

```bash
git clone https://github.com/PheaRak77/handgesture-recognition.git
cd handgesture-recognition
```

### Step 3: Create & Activate Virtual Environment

- **Linux / macOS**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- **Windows (Command Prompt / PowerShell)**:
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Usage Guide

### Step 1: Discover Connected Cameras

Scan available video inputs (USB webcams, integrated cameras, or DroidCam WiFi streams):

```bash
python scripts/find_cameras.py
```

To test a mobile camera via DroidCam WiFi:
```bash
python scripts/find_cameras.py --ip 192.168.1.15
```

### Step 2: Run Live Gesture Recognition

Run recognition using your primary camera (default index 0):

```bash
python main.py
```

To specify a specific camera index (e.g., camera 1):
```bash
python main.py --camera 1
```

To scan multiple camera indices (0–3) automatically:
```bash
python main.py --scan-cameras
```

### Step 3: Run Demo Mode with Video File

Test gesture recognition using a recorded video file:

```bash
python main.py --demo assets/sample_video.mp4
```

### Step 4: Preview Supported Gesture Reference

Launch the interactive GUI catalog showing all registered gestures:

```bash
python demo_gestures.py
```

---

## 🧠 Step-by-Step Machine Learning Training Guide

You can train a custom Machine Learning model (`RandomForestClassifier`) using your own hand landmark dataset for increased gesture accuracy.

### Step 1: Collect Landmark Data

Run the interactive data collector script:

```bash
python scripts/collect_landmarks.py --camera 0
```

**Controls while collecting:**
- `[` / `]`: Switch target gesture label
- `SPACE`: Record 30 consecutive frames for the active gesture label
- `n`: Select `No Hand` label
- `q`: Save and exit collector

*Recommended: Collect 50+ samples per gesture.*

### Step 2: Train Classifier Model

Execute the training script to process CSV files from `data/landmarks/` and output `models/gesture_classifier.joblib`:

```bash
python scripts/train_classifier.py
```

### Step 3: Run with Trained Model

Once `models/gesture_classifier.joblib` exists, `main.py` automatically detects and uses the ML model for classification alongside rule-based fallbacks.

---

## 🗄️ Step-by-Step Database Setup & Logging (PostgreSQL)

Logging gesture events and session metrics to PostgreSQL is optional.

### Step 1: Configure Environment Variables

1. Copy the example configuration file:
   ```bash
   cp env.example .env
   ```
2. Edit `.env` with your database credentials:
   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=db_num_project
   DB_USER=postgres
   DB_PASSWORD=your_password
   ENABLE_DB=true
   ```

### Step 2: Initialize Database Schema

Execute the SQL script in pgAdmin or psql:
```bash
psql -U postgres -d db_num_project -f database/02_seed_gestures.sql
```

### Step 3: Verify Connection

```bash
python scripts/test_db.py
```

### Step 4: Run Application with Database Logging

```bash
python main.py --db
```

---

## ⚙️ Camera Configuration (`config/cameras.yaml`)

Customize smoothing frames, ML confidence thresholds, and camera mirroring per camera index:

```yaml
defaults:
  min_phrase_frames: 2    # Confirmed frames (~66ms at 30fps)
  min_hold_frames: 2      # Frames with no hand before phrase clears
  ml_confidence: 0.55     # Minimum ML confidence score (0.0 - 1.0)
  mirror: false
  use_ml: true            # Use ML classifier when available

cameras:
  "0":
    mirror: false
    min_phrase_frames: 2
  "1":
    mirror: true          # Recommended for front-facing / phone camera
    min_phrase_frames: 2
```

---

## 🎮 Controls & Keyboard Shortcuts

| Shortcut | Context | Action |
| :--- | :--- | :--- |
| `q` or `ESC` | Live Window / Demo Window | Focus OpenCV window and press to quit app gracefully |
| `Ctrl + C` | Terminal | Gracefully terminate all active background threads |
| `[` / `]` | `collect_landmarks.py` | Change active gesture label |
| `SPACE` | `collect_landmarks.py` | Record landmark batch (30 frames) |

---

## ❓ Troubleshooting & FAQ

- **OpenCV window won't close on keypress**: Click directly on the video output window to give it focus before pressing `q` or `ESC`.
- **No working camera detected**: Check USB cables or run `python scripts/find_cameras.py` to confirm assigned camera indices.
- **Low FPS or camera lag**: Ensure you are running Python with standard performance settings. The multi-threaded pipeline automatically balances inference work across CPU cores.
- **Database connection failure**: Check if PostgreSQL server is running and `.env` credentials match your local database configuration.