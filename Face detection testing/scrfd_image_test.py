import cv2
import numpy as np
from insightface.app import FaceAnalysis

def main():
    # Load model (SCRFD + ArcFace), CPU only
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # Load your input image
    img = cv2.imread("test.jpg")
    if img is None:
        print("Image not found. Check your path.")
        return

    # Run face detection
    faces = app.get(img)
    print(f"Detected {len(faces)} face(s).")

    # Draw bounding boxes
    for f in faces:
        box = f.bbox.astype(int)
        x1, y1, x2, y2 = box

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),  # green box
            2
        )

        # Optional: write confidence score on the box
        cv2.putText(img, "face", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

    # Save the output image
    output_path = "output_detected.jpg"
    cv2.imwrite(output_path, img)

    print(f"Output saved as {output_path}")

    # Display the result
    cv2.imshow("Detected Faces (CPU)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
