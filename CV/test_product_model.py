from ultralytics import YOLO
from pathlib import Path
import cv2

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Your trained model
MODEL_PATH = BASE_DIR / "runs" / "retail" / "retail_products_v3" / "weights" / "best.pt"

# Test image
IMAGE_PATH = BASE_DIR / "testimage" /"WhatsApp Image 2026-09-23 at 9.33.43 PM (2).jpeg"

# Load trained model
model = YOLO(str(MODEL_PATH))

print("Model loaded:")
print(MODEL_PATH)

# Run prediction
results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.5,
    imgsz=640,
    save=True
)

# Display result
for result in results:
    annotated = result.plot()

    cv2.imshow("RetailEdge AI - Product Detection", annotated)

    print("\nDetected products:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        product_name = model.names[class_id]

        print(
            f"{product_name} : "
            f"{confidence:.2f}"
        )

cv2.waitKey(0)
cv2.destroyAllWindows()