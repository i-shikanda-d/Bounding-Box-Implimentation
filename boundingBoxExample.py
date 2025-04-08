import cv2

# Load the image
image = cv2.imread("./sample2.jpg")  # Replace with your image

# Convert to grayscale (faster & better for detection)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,  # Adjust to detect different face sizes
    minNeighbors=5,    # Higher = fewer false positives
    minSize=(30, 30)   # Minimum face size
)

# Draw green boxes around faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box, thickness=2

# Display the result
cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()