import cv2
import os
import time
from insightface.app import FaceAnalysis

print(">>> SCRIPT STARTED <<<")

# 1. Setup save directory
save_dir = "captured_faces"
print("Save directory:", save_dir)
os.makedirs(save_dir, exist_ok=True)

# 2. Load the model
print("Loading model...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))
print("Model loaded.")

# 3. Open webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("cap.isOpened():", cap.isOpened())

if not cap.isOpened():
    print("ERROR: Webcam not opened.")
    exit()

print("Webcam opened successfully.")

# 4. Main loop
while True:
    ret, frame = cap.read()
    print("Frame read:", ret)

    if not ret:
        break

    faces = app.get(frame)
    print("Faces detected:", len(faces))

    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)

        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = frame[y1:y2, x1:x2]

        print("Crop size:", crop.shape if crop.size > 0 else "EMPTY")

        if crop.size > 0:
            crop_resized = cv2.resize(crop, (112, 112))
            filename = f"{save_dir}/face_{int(time.time()*1000)}.jpg"
            ok = cv2.imwrite(filename, crop_resized)
            print("Saving:", filename, "| Success:", ok)

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow("TEST", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# 5. Cleanup
print("Closing...")
cap.release()
cv2.destroyAllWindows()
print(">>> SCRIPT ENDED <<<")
