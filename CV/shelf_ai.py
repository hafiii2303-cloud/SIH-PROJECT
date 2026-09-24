import cv2
import requests
from collections import defaultdict, deque
from ultralytics import YOLO
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "retail"
    / "retail_products_v3-2"
    / "weights"
    / "best.pt"
)

CAMERA_INDEX = 0

API_URL = "http://127.0.0.1:8000/api/shelf-event"


# =========================================================
# ACCURACY SETTINGS
# =========================================================

CONFIDENCE_THRESHOLD = 0.50

INFERENCE_CONFIDENCE = 0.40

IOU_THRESHOLD = 0.50

IMAGE_SIZE = 960

MAX_DETECTIONS = 100


# =========================================================
# TEMPORAL CONFIRMATION
# =========================================================

# Product must be detected several times
# before we consider it stable.

CONFIRMATION_FRAMES = 5

COOLDOWN_FRAMES = 90


# =========================================================
# SHELF REGIONS
# =========================================================

SHELF_REGIONS = {

    "SHELF_A":
        (20, 30, 400, 220),

    "SHELF_B":
        (420, 30, 820, 220),

    "SHELF_C":
        (20, 240, 400, 450)

}


# =========================================================
# PRODUCT INFORMATION
#
# MUST MATCH data.yaml CLASS IDs
# =========================================================

PRODUCTS = {

    0: {
        "name": "Sprite",
        "barcode": "890100000003",
        "expected_shelf": "SHELF_B"
    },

    1: {
        "name": "Coca Cola",
        "barcode": "890100000006",
        "expected_shelf": "SHELF_B"
    },

    2: {
        "name": "Good Day",
        "barcode": "890100000002",
        "expected_shelf": "SHELF_A"
    },

    3: {
        "name": "Green Chips",
        "barcode": "890100000001",
        "expected_shelf": "SHELF_A"
    },

    4: {
        "name": "Green Milk",
        "barcode": "890100000005",
        "expected_shelf": "SHELF_B"
    },

    5: {
        "name": "Lifebuoy",
        "barcode": "890100000004",
        "expected_shelf": "SHELF_C"
    }

}


# =========================================================
# FIND SHELF
# =========================================================

def find_shelf(x, y):

    for shelf_name, (
        x1,
        y1,
        x2,
        y2
    ) in SHELF_REGIONS.items():

        if (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        ):

            return shelf_name

    return None


# =========================================================
# SEND ALERT
# =========================================================

def send_shelf_event(
    barcode,
    detected_shelf
):

    payload = {

        "trolley_id":
            "SHELF_CAMERA_01",

        "barcode":
            barcode,

        "detected_shelf":
            detected_shelf

    }

    try:

        response = requests.post(

            API_URL,

            json=payload,

            timeout=1.5

        )

        response.raise_for_status()

        print(
            f"Alert sent: {barcode} -> {detected_shelf}"
        )

    except requests.RequestException as error:

        print(
            "Shelf API error:",
            error
        )


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading model...")

model = YOLO(
    str(MODEL_PATH)
)

print("Model loaded successfully.")


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open shelf camera."
    )


# =========================================================
# DETECTION HISTORY
#
# Stores recent detections for each product.
# =========================================================

detection_history = {

    class_id: deque(
        maxlen=CONFIRMATION_FRAMES
    )

    for class_id in PRODUCTS
}


# =========================================================
# ALERT COOLDOWN
# =========================================================

cooldowns = defaultdict(int)

frame_number = 0


# =========================================================
# MAIN LOOP
# =========================================================

try:

    while True:

        success, frame = cap.read()

        if not success:

            print(
                "Could not read camera frame."
            )

            break


        # -------------------------------------------------
        # Resize
        # -------------------------------------------------

        frame = cv2.resize(
            frame,
            (852, 480)
        )


        # -------------------------------------------------
        # Draw shelf regions
        # -------------------------------------------------

        for shelf_name, (
            x1,
            y1,
            x2,
            y2
        ) in SHELF_REGIONS.items():

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (255, 180, 0),

                2

            )

            cv2.putText(

                frame,

                shelf_name,

                (x1 + 5, y1 + 22),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.60,

                (255, 180, 0),

                2

            )


        # =================================================
        # YOLO DETECTION
        # =================================================

        results = model.predict(

            frame,

            conf=INFERENCE_CONFIDENCE,

            iou=IOU_THRESHOLD,

            imgsz=IMAGE_SIZE,

            max_det=MAX_DETECTIONS,

            verbose=False

        )


        # Track which products appeared
        # in this frame.

        current_detections = set()


        # =================================================
        # PROCESS DETECTIONS
        # =================================================

        for result in results:

            if result.boxes is None:

                continue


            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )


                # -------------------------------------------------
                # Final confidence filtering
                # -------------------------------------------------

                if confidence < CONFIDENCE_THRESHOLD:

                    continue


                class_id = int(
                    box.cls[0]
                )


                product = PRODUCTS.get(
                    class_id
                )


                if product is None:

                    continue


                current_detections.add(
                    class_id
                )


                # -------------------------------------------------
                # Bounding box
                # -------------------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )


                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )


                # -------------------------------------------------
                # Find shelf
                # -------------------------------------------------

                detected_shelf = find_shelf(

                    center_x,

                    center_y

                )


                # -------------------------------------------------
                # Check shelf
                # -------------------------------------------------

                if detected_shelf is None:

                    status = (
                        "OUTSIDE REGION"
                    )

                elif (
                    detected_shelf
                    !=
                    product[
                        "expected_shelf"
                    ]
                ):

                    status = "MISPLACED"

                else:

                    status = "CORRECT"


                # -------------------------------------------------
                # Store confidence
                # -------------------------------------------------

                detection_history[
                    class_id
                ].append(
                    confidence
                )


                # -------------------------------------------------
                # Calculate average confidence
                # -------------------------------------------------

                history = (
                    detection_history[
                        class_id
                    ]
                )


                average_confidence = (

                    sum(history)
                    /
                    len(history)

                )


                # -------------------------------------------------
                # Confirm detection
                # -------------------------------------------------

                confirmed = (

                    len(history)
                    >=
                    CONFIRMATION_FRAMES

                    and

                    average_confidence
                    >=
                    CONFIDENCE_THRESHOLD

                )


                # -------------------------------------------------
                # Colors
                # -------------------------------------------------

                if status == "MISPLACED":

                    box_color = (
                        0,
                        0,
                        255
                    )

                elif confirmed:

                    box_color = (
                        0,
                        255,
                        0
                    )

                else:

                    box_color = (
                        0,
                        255,
                        255
                    )


                # -------------------------------------------------
                # Draw box
                # -------------------------------------------------

                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    box_color,

                    2

                )


                # -------------------------------------------------
                # Label
                # -------------------------------------------------

                label = (

                    f"{product['name']} "

                    f"{confidence:.2f} "

                    f"{status}"

                )


                cv2.putText(

                    frame,

                    label,

                    (
                        x1,
                        max(
                            20,
                            y1 - 8
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.50,

                    box_color,

                    2

                )


                # -------------------------------------------------
                # Confirmation text
                # -------------------------------------------------

                if confirmed:

                    confirmation_text = (
                        "CONFIRMED"
                    )

                else:

                    confirmation_text = (
                        "VERIFYING..."
                    )


                cv2.putText(

                    frame,

                    confirmation_text,

                    (
                        x1,
                        min(
                            470,
                            y2 + 18
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    box_color,

                    1

                )


                # =================================================
                # SEND MISPLACEMENT ALERT
                # =================================================

                if (

                    confirmed

                    and

                    status
                    ==
                    "MISPLACED"

                    and

                    detected_shelf
                    is not None

                    and

                    frame_number
                    >=
                    cooldowns[
                        product["barcode"]
                    ]

                ):

                    send_shelf_event(

                        product[
                            "barcode"
                        ],

                        detected_shelf

                    )


                    cooldowns[
                        product["barcode"]
                    ] = (

                        frame_number
                        +
                        COOLDOWN_FRAMES

                    )


        # =================================================
        # DISPLAY INFORMATION
        # =================================================

        cv2.putText(

            frame,

            f"Frame: {frame_number}",

            (10, 25),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Image Size: {IMAGE_SIZE}",

            (10, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            (255, 255, 255),

            1

        )


        cv2.imshow(

            "RetailEdge AI - Shelf AI",

            frame

        )


        # =================================================
        # KEYBOARD
        # =================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q") or key == 27:

            break


        frame_number += 1


finally:

    cap.release()

    cv2.destroyAllWindows()