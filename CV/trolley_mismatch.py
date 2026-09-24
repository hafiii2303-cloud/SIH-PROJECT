from ultralytics import YOLO
from pathlib import Path
from collections import deque
import cv2
import requests
import time


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "retail"
    / "retail_products_v3-2"
    / "weights"
    / "best.pt"
)

API_URL = "http://127.0.0.1:8000/api/product-mismatch"

CAMERA_INDEX = 0


# ============================================================
# YOLO SETTINGS
# ============================================================

INFERENCE_CONFIDENCE = 0.40
FINAL_CONFIDENCE = 0.50

IMAGE_SIZE = 960
MAX_DETECTIONS = 20
IOU_THRESHOLD = 0.50


# ============================================================
# CONFIRMATION SETTINGS
# ============================================================

CONFIRMATION_FRAMES = 5
AVERAGE_CONFIDENCE_THRESHOLD = 0.55

ALERT_COOLDOWN = 90


# ============================================================
# PRODUCT INFORMATION
# ============================================================
# IMPORTANT:
# This mapping MUST match your data.yaml
#
# 0 Sprite
# 1 Coca Cola
# 2 Good Day
# 3 Green Chips
# 4 Green Milk
# 5 Lifebuoy
# ============================================================

PRODUCTS = {
    0: {
        "name": "Sprite",
        "barcode": "890100000003"
    },

    1: {
        "name": "Coca Cola",
        "barcode": "890100000006"
    },

    2: {
        "name": "Good Day",
        "barcode": "890100000002"
    },

    3: {
        "name": "Green Chips",
        "barcode": "890100000001"
    },

    4: {
        "name": "Green Milk",
        "barcode": "890100000005"
    },

    5: {
        "name": "Lifebuoy",
        "barcode": "890100000004"
    }
}


# ============================================================
# SEND MISMATCH TO BACKEND
# ============================================================

def send_mismatch(
    trolley_id,
    expected_barcode,
    detected_barcode,
    confidence
):

    data = {
        "trolley_id": trolley_id,
        "expected_barcode": expected_barcode,
        "detected_barcode": detected_barcode,
        "confidence": confidence
    }

    try:

        response = requests.post(
            API_URL,
            json=data,
            timeout=2
        )

        if response.ok:

            print("\n--------------------------------")
            print("MISMATCH SENT TO BACKEND")
            print("--------------------------------")
            print(response.json())
            print("--------------------------------\n")

        else:

            print(
                "Backend returned:",
                response.status_code
            )

    except requests.exceptions.RequestException as error:

        print("Backend connection error:")
        print(error)


# ============================================================
# GET PRODUCT NAME FROM BARCODE
# ============================================================

def get_product_name(barcode):

    for product in PRODUCTS.values():

        if product["barcode"] == barcode:

            return product["name"]

    return "Unknown Product"


# ============================================================
# LOAD MODEL
# ============================================================

print("\n========================================")
print("RetailEdge AI - Smart Trolley")
print("Product Verification")
print("========================================")

print("Model path:")
print(MODEL_PATH)

if not MODEL_PATH.exists():

    print("\nERROR: Model file not found!")
    print(MODEL_PATH)

    raise SystemExit


print("\nLoading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")


# ============================================================
# TROLLEY INFORMATION
# ============================================================

TROLLEY_ID = "TROLLEY_01"


# ============================================================
# EXPECTED PRODUCT
# ============================================================

print("\n----------------------------------------")
print("SMART TROLLEY")
print("----------------------------------------")

expected_barcode = input(
    "Enter scanned product barcode: "
).strip()

expected_product = get_product_name(
    expected_barcode
)

print("\nExpected product:")
print(expected_product)

print("Expected barcode:")
print(expected_barcode)

print("----------------------------------------")


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    print("\nERROR: Cannot open camera.")

    raise SystemExit


# ============================================================
# PREDICTION HISTORY
# ============================================================

recent_predictions = deque(
    maxlen=CONFIRMATION_FRAMES
)

recent_confidences = deque(
    maxlen=CONFIRMATION_FRAMES
)


# ============================================================
# VARIABLES
# ============================================================

frame_number = 0

last_alert_frame = -9999

status = "WAITING FOR PRODUCT"

detected_product_name = "None"

detected_barcode = ""

detected_confidence = 0.0

confirmed_product = None


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Camera frame not received.")

        break


    frame_number += 1


    # --------------------------------------------------------
    # RESIZE FRAME
    # --------------------------------------------------------

    frame = cv2.resize(
        frame,
        (960, 540)
    )


    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model(
        frame,
        conf=INFERENCE_CONFIDENCE,
        imgsz=IMAGE_SIZE,
        iou=IOU_THRESHOLD,
        max_det=MAX_DETECTIONS,
        verbose=False
    )


    # --------------------------------------------------------
    # RESET CURRENT DETECTION
    # --------------------------------------------------------

    best_detection = None


    # --------------------------------------------------------
    # FIND BEST PRODUCT
    # --------------------------------------------------------

    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue


        for box in boxes:

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )


            # Ignore unknown classes

            if class_id not in PRODUCTS:
                continue


            # Ignore low confidence

            if confidence < FINAL_CONFIDENCE:
                continue


            # Keep highest confidence detection

            if (
                best_detection is None
                or confidence > best_detection["confidence"]
            ):

                best_detection = {

                    "class_id": class_id,

                    "confidence": confidence,

                    "box": box.xyxy[0].cpu().numpy()
                }


    # ========================================================
    # IF PRODUCT DETECTED
    # ========================================================

    if best_detection is not None:

        class_id = best_detection["class_id"]

        confidence = best_detection["confidence"]

        product = PRODUCTS[class_id]

        product_name = product["name"]

        barcode = product["barcode"]


        # ----------------------------------------------------
        # SAVE PREDICTION
        # ----------------------------------------------------

        recent_predictions.append(
            class_id
        )

        recent_confidences.append(
            confidence
        )


        # ----------------------------------------------------
        # DRAW BOUNDING BOX
        # ----------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            best_detection["box"]
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        label = (
            f"{product_name} "
            f"{confidence:.2f}"
        )


        cv2.putText(
            frame,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # CONFIRM PRODUCT
        # ----------------------------------------------------

        if len(recent_predictions) >= CONFIRMATION_FRAMES:

            same_product = (
                len(set(recent_predictions)) == 1
            )

            average_confidence = (
                sum(recent_confidences)
                / len(recent_confidences)
            )


            if (
                same_product
                and
                average_confidence
                >= AVERAGE_CONFIDENCE_THRESHOLD
            ):

                confirmed_product = (
                    recent_predictions[-1]
                )


                confirmed_product_data = PRODUCTS[
                    confirmed_product
                ]


                detected_product_name = (
                    confirmed_product_data["name"]
                )

                detected_barcode = (
                    confirmed_product_data["barcode"]
                )

                detected_confidence = (
                    average_confidence
                )


                # =================================================
                # COMPARE EXPECTED VS DETECTED
                # =================================================

                if detected_barcode == expected_barcode:

                    status = "MATCH"


                else:

                    status = "MISMATCH"


                    # ---------------------------------------------
                    # SEND ALERT
                    # ---------------------------------------------

                    if (
                        frame_number
                        - last_alert_frame
                        >= ALERT_COOLDOWN
                    ):

                        print("\n================================")
                        print("PRODUCT MISMATCH DETECTED")
                        print("================================")

                        print(
                            "Trolley:",
                            TROLLEY_ID
                        )

                        print(
                            "Expected:",
                            expected_product
                        )

                        print(
                            "Detected:",
                            detected_product_name
                        )

                        print(
                            "Expected barcode:",
                            expected_barcode
                        )

                        print(
                            "Detected barcode:",
                            detected_barcode
                        )

                        print(
                            "Confidence:",
                            f"{detected_confidence:.2f}"
                        )

                        print(
                            "================================\n"
                        )


                        send_mismatch(
                            TROLLEY_ID,
                            expected_barcode,
                            detected_barcode,
                            detected_confidence
                        )


                        last_alert_frame = (
                            frame_number
                        )


    else:

        # ----------------------------------------------------
        # NO PRODUCT
        # ----------------------------------------------------

        status = "WAITING FOR PRODUCT"

        detected_product_name = "None"

        detected_barcode = ""

        detected_confidence = 0.0

        recent_predictions.clear()

        recent_confidences.clear()


    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    # Expected product

    cv2.putText(
        frame,
        f"Expected: {expected_product}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # Expected barcode

    cv2.putText(
        frame,
        f"Expected Barcode: {expected_barcode}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # Detected product

    cv2.putText(
        frame,
        f"Detected: {detected_product_name}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # Detected barcode

    if detected_barcode:

        cv2.putText(
            frame,
            f"Detected Barcode: {detected_barcode}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    # Confidence

    cv2.putText(
        frame,
        f"Confidence: {detected_confidence:.2f}",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # Confirmation count

    cv2.putText(
        frame,
        f"Confirmation: "
        f"{len(recent_predictions)}/{CONFIRMATION_FRAMES}",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # STATUS COLOR
    # ========================================================

    if status == "MATCH":

        status_color = (
            0,
            255,
            0
        )

    elif status == "MISMATCH":

        status_color = (
            0,
            0,
            255
        )

    else:

        status_color = (
            0,
            255,
            255
        )


    # ========================================================
    # STATUS DISPLAY
    # ========================================================

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (20, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        status_color,
        3
    )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "RetailEdge AI - Smart Trolley Verification",
        (20, 520),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "RetailEdge AI - Trolley Verification",
        frame
    )


    # ========================================================
    # EXIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\nSmart Trolley verification stopped.")