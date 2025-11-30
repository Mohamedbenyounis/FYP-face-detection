# FYP Face Detection & Recognition System

A face detection and recognition system using InsightFace (SCRFD + ArcFace) for real-time video and static image processing.

## Features

- **Face Detection**: Uses SCRFD model for accurate face detection
- **Face Recognition**: Uses ArcFace embeddings for face matching
- **Real-time Recognition**: Process video streams from webcam
- **Static Image Recognition**: Process and analyze single images
- **User Registration**: Register new faces and store embeddings
- **Confidence Scoring**: Displays similarity scores for face matches
- **Alert System**: Plays audio alert when unauthorised faces are detected
- **Event Logging**: JSON logs of all detection/recognition events

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Mohamedbenyounis/FYP-face-detection.git
cd FYP-face-detection
```

### 2. Create Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv fyp_env
fyp_env\Scripts\activate
```

**On Windows (Command Prompt):**
```cmd
python -m venv fyp_env
fyp_env\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
python3 -m venv fyp_env
source fyp_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install opencv-python
pip install insightface
pip install scikit-learn
pip install numpy
pip install onnxruntime
pip install playsound
```

**Note**: If you encounter issues with `playsound`, you may need to install it separately or use an alternative audio library.

### 4. Download InsightFace Models

The models will be downloaded automatically on first run. Make sure you have an internet connection.

---

## Usage

### Option 1: Real-Time Video Recognition (`main.py`)

This script provides real-time face detection and recognition using your webcam.

#### Run the script:
```bash
python main.py
```

#### Menu Options:
- **1** - Register a new face
  - Enter user name
  - Provide path to registration image
  - Face embedding will be saved to `registered_faces/`

- **2** - Run real-time recognition
  - Opens webcam feed
  - Detects and recognizes faces in real-time
  - Displays FPS counter
  - Press `q` to quit

- **q** - Quit the program

#### Example Registration:
```
Choose: 1
Enter user name: John
Path to image: img/john.jpg
```

---

### Option 2: Static Image Recognition (`Face detection testing/scrfd_image_test.py`)

This script processes single images for face detection and recognition.

#### Navigate to the testing directory:
```bash
cd "Face detection testing"
```

#### Run the script:
```bash
python scrfd_image_test.py
```

#### Menu Options:
- **1** - Register a new face
  - Enter user name
  - Provide path to registration image
  - Face embedding will be saved to `registered_faces/`

- **2** - Run recognition on image
  - Enter image path
  - Processes image and saves output with bounding boxes and labels
  - Generates JSON log in `logs/event_log.json`
  - Plays alert sound if unauthorised faces are detected

- **q** - Quit the program

#### Example Usage:
```
Choose: 2
Enter path to image: img/group_photo.jpg
```

#### Alert System:
- An audio alert (`Alert/alert-33762.mp3`) automatically plays when unauthorised or unknown faces are detected
- Ensure the alert sound file is present in the `Alert/` directory

#### Output:
- **Image**: `output_img/output_recognized_YYYY-MM-DD_HH-MM-SS.jpg`
  - Green boxes = Recognized users
  - Red boxes = Unauthorised/Unknown faces
  - Labels show name and confidence score

- **Log**: `logs/event_log.json`
  ```json
  {
    "timestamp": "2025-11-30_14-30-45",
    "source_image": "test1.jpg",
    "face_index": 0,
    "bbox": [100, 150, 300, 350],
    "det_score": 0.99,
    "recognized_as": "Mohamed",
    "confidence_score": 0.78,
    "event_type": "face_recognized"
  }
  ```

---

## Project Structure

```
FYP-face-detection/
│
├── main.py                          # Real-time video recognition
├── registered_faces/                # Stored face embeddings (.npy files)
│   ├── Mohamed.npy
│   └── test1.npy
│
├── Face detection testing/
│   ├── scrfd_image_test.py         # Static image recognition
│   ├── registered_faces/           # Stored face embeddings for testing
│   ├── logs/
│   │   └── event_log.json          # Detection/recognition logs
│   ├── output_img/                 # Output images with annotations
│   ├── img/                        # Input test images
│   └── Alert/
│       └── alert-33762.mp3         # Audio alert for unauthorised detection
│
├── fyp_env/                        # Virtual environment (not committed)
├── img/                            # Additional images
└── README.md                       # This file
```

---

## Recognition Threshold

The default confidence threshold is **0.45** (cosine similarity score).

- **Score > 0.45**: Face is recognized and labeled with user name
- **Score ≤ 0.45**: Face is labeled as "Unauthorised"

To adjust the threshold, modify the `threshold` parameter in the `match_face()` function:

```python
def match_face(frame_emb, registered, threshold=0.45):
```

---

## Troubleshooting

### Virtual Environment Not Activating
- **Windows PowerShell**: You may need to enable script execution:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### Camera Not Opening
- Ensure no other application is using the webcam
- Try changing camera index in `cv2.VideoCapture(0)` to `1` or `2`

### No Face Detected During Registration
- Ensure the image contains a clear, front-facing face
- Use good lighting conditions
- Image should be in a supported format (JPG, PNG)

### Import Errors
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

### Model Download Issues
- Ensure you have an active internet connection
- Models are downloaded to `~/.insightface/models/`

### Alert Sound Not Playing
- Ensure `playsound` is installed: `pip install playsound`
- Verify the alert sound file exists at `Face detection testing/Alert/alert-33762.mp3`
- Check that your system audio is not muted

---

## Requirements

- Python 3.8+
- Webcam (for real-time recognition)
- Windows/macOS/Linux

---

## License

This project is for academic purposes (Final Year Project).

---

## Author

**Mohamed Ben Younis**  
Repository: [FYP-face-detection](https://github.com/Mohamedbenyounis/FYP-face-detection)
