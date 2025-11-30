import cv2
import json
import numpy as np
import os
from datetime import datetime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from playsound import playsound
import pygame


LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.path.join(LOG_DIR, "event_log.json")
REGISTER_DIR = os.path.join(os.path.dirname(__file__), "registered_faces")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_img")
ALERT_SOUND = os.path.join(os.path.dirname(__file__), "Alert", "alert-33762.mp3")


def log_event(event_data):
    """Append face detection event to JSON log."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(event_data)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

    print(f"[LOG] Event saved → {LOG_FILE}")


def load_registered():
    """Load registered face embeddings from the registered_faces directory."""
    users = {}
    
    if not os.path.exists(REGISTER_DIR):
        print(f"[WARNING] No registered faces directory found at {REGISTER_DIR}")
        return users
    
    for file in os.listdir(REGISTER_DIR):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            emb = np.load(os.path.join(REGISTER_DIR, file))
            users[name] = emb
    
    print(f"[INFO] Loaded {len(users)} registered user(s): {list(users.keys())}")
    return users


def match_face(frame_emb, registered, threshold=0.45):
    """Match detected face embedding against registered users."""
    best_user = None
    best_score = -1

    for name, emb in registered.items():
        score = cosine_similarity([frame_emb], [emb])[0][0]
        if score > best_score:
            best_score = score
            best_user = name

    if best_score > threshold:
        return best_user, best_score
    else:
        return "Unauthorised", best_score


def play_alert():
    """Play alert sound for unauthorised detection."""
    try:
        if os.path.exists(ALERT_SOUND):
            print("[ALERT] Unauthorised face detected! Playing alert sound...")
            pygame.mixer.init()
            pygame.mixer.music.load(ALERT_SOUND)
            pygame.mixer.music.play()
            # Wait for the sound to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.quit()
        else:
            print(f"[WARNING] Alert sound not found at {ALERT_SOUND}")
    except Exception as e:
        print(f"[ERROR] Could not play alert sound: {e}")


def register_user(app, name, img_path):
    """Register a new user by extracting and saving face embedding."""
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"[ERROR] Image not found at {img_path}")
        return
    
    faces = app.get(img)

    if len(faces) == 0:
        print("[ERROR] No face detected in registration image")
        return

    emb = faces[0].embedding
    
    # Ensure directory exists
    os.makedirs(REGISTER_DIR, exist_ok=True)
    
    save_path = os.path.join(REGISTER_DIR, f"{name}.npy")
    np.save(save_path, emb)
    print(f"[SUCCESS] Registered: {name} → {save_path}")


def run_recognition(app):
    """Run face detection and recognition on a single image."""
    # Load registered face embeddings
    registered = load_registered()

    # Input image path
    img_path = input("Enter path to image: ").strip()
    
    img = cv2.imread(img_path)

    if img is None:
        print(f"[ERROR] Image not found at {img_path}")
        return

    print(f"[INFO] Running detection and recognition on {img_path} …")

    faces = app.get(img)
    num_faces = len(faces)

    print(f"[INFO] Faces detected: {num_faces}")

    # Timestamp for logs & output file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Flag to track if any unauthorised face is detected
    unauthorised_detected = False

    # Process each face
    for idx, f in enumerate(faces):
        x1, y1, x2, y2 = f.bbox.astype(int)
        emb = f.embedding

        # Perform face recognition
        if len(registered) > 0:
            label, confidence = match_face(emb, registered)
        else:
            label = "Unknown"
            confidence = 0.0

        # Choose color based on recognition result
        if label != "Unauthorised" and label != "Unknown":
            color = (0, 255, 0)  # Green for recognized
        else:
            color = (0, 0, 255)  # Red for unauthorised/unknown
            unauthorised_detected = True

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label with confidence score
        text = f"{label} ({confidence:.2f})"
        
        # Add background rectangle for text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 10), 
                     (x1 + text_size[0], y1), color, -1)
        
        cv2.putText(img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        # Create metadata for each detected face
        event_data = {
            "timestamp": timestamp,
            "source_image": img_path,
            "face_index": idx,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "det_score": float(f.det_score),
            "recognized_as": label,
            "confidence_score": float(confidence),
            "event_type": "face_recognized" if label not in ["Unknown", "Unauthorised"] else "face_unauthorised"
        }

        log_event(event_data)

    # Play alert if any unauthorised face was detected
    if unauthorised_detected:
        play_alert()

    # Save output image
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"output_recognized_{timestamp}.jpg")
    cv2.imwrite(output_path, img)
    print(f"[INFO] Processed image saved → {output_path}")

    # Show result
    cv2.imshow("Face Detection & Recognition", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("[INFO] Loading face detection and recognition model…")

    # Load FaceAnalysis model
    app = FaceAnalysis("buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))

    print("\n=== Face Recognition System ===")
    print("1 = Register new face")
    print("2 = Run recognition on image")
    print("q = Quit")
    choice = input("Choose: ").strip()

    if choice == "1":
        name = input("Enter user name: ").strip()
        path = input("Path to registration image: ").strip()
        register_user(app, name, path)

    elif choice == "2":
        run_recognition(app)

    elif choice == 'q':
        cv2.destroyAllWindows()
        print("[INFO] Exiting…")
    
    else:
        print("[ERROR] Invalid choice")


if __name__ == "__main__":
    main()
