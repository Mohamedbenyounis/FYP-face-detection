import cv2
import json
import time
from datetime import datetime
from insightface.app import FaceAnalysis
import os


LOG_FILE = "event_log.json"


def log_event(event_data):
    """Append face detection event to JSON log."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(event_data)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

    print(f"[LOG] Event saved → {LOG_FILE}")


def main():
    print("[INFO] Loading face detection model…")

    app = FaceAnalysis("buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))

    img_path = "test1.jpg"
    img = cv2.imread(img_path)

    if img is None:
        print("[ERROR] Image not found.")
        return

    print(f"[INFO] Running detection on {img_path} …")

    faces = app.get(img)
    num_faces = len(faces)

    print(f"[INFO] Faces detected: {num_faces}")

    # Timestamp for logs & output file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Process each face
    for idx, f in enumerate(faces):
        x1, y1, x2, y2 = f.bbox.astype(int)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Create metadata for each detected face
        event_data = {
            "timestamp": timestamp,
            "source_image": img_path,
            "face_index": idx,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "det_score": float(f.det_score),
            "event_type": "face_detected"
        }

        log_event(event_data)

    # Save output image
    output_path = f"output_detected_{timestamp}.jpg"
    cv2.imwrite(output_path, img)
    print(f"[INFO] Processed image saved → {output_path}")

    # Show result
    cv2.imshow("Face Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
