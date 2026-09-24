from ultralytics import YOLO
from pathlib import Path
import cv2

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Your trained model
MODEL_PATH = BASE_DIR / "runs" / "retail" / "retail_products_v3-2" / "weights" / "best.pt"

# Load trained model
model = YOLO(str(MODEL_PATH))

print("Model loaded:")
print(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")

while True:

    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO prediction on current frame
    results = model.predict(
        source=frame,
        conf=0.65,
        imgsz=640,
        verbose=False
    )

    # Get first result
    result = results[0]

    # Draw bounding boxes
    annotated = result.plot()

    # Print detected products
    detected_products = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        product_name = model.names[class_id]

        detected_products.append(
            f"{product_name} ({confidence:.2f})"
        )

    # Print detections
    if detected_products:
        print("Detected:", ", ".join(detected_products))

    # Show webcam
    cv2.imshow(
        "RetailEdge AI - Product Detection",
        annotated
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()