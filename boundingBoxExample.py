import cv2
import numpy as np
import os

# ===== 1. Load Image =====
image_path = "./sample.jpg"
if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found!")
    exit()

image = cv2.imread(image_path)
if image is None:
    print(f"Error: Failed to load '{image_path}'. Check file format or corruption.")
    exit()

# ===== 2. Prepare Outputs =====
# Left Panel: Contours only (no text)
contour_output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contour_output, contours, -1, (255, 0, 0), 2)  # Just blue contours

# Right Panel: Faces with coordinates
face_output = image.copy()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    x2, y2 = x + w, y + h
    cv2.rectangle(face_output, (x, y), (x2, y2), (0, 255, 0), 2)  # Green box
    cv2.putText(face_output, f"({x},{y})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)  # Top-left
    cv2.putText(face_output, f"({x2},{y2})", (x2-50, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)  # Bottom-right
    cv2.putText(face_output, f"W:{w} H:{h}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)  # Size

# ===== 3. Combine & Label =====
contour_output = cv2.putText(contour_output, "CONTOURS", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
face_output = cv2.putText(face_output, "FACES", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
combined = np.hstack((contour_output, face_output))

# ===== 4. Display/Save =====
cv2.imshow("Contours (Left) vs Faces with Coordinates (Right)", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("contours_vs_faces.jpg", combined)  # Optional save