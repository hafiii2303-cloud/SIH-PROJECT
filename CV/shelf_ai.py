import cv2
import requests
from collections import defaultdict
from ultralytics import YOLO


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "models/retail_products.pt"

CAMERA_INDEX = 0

API_URL = (
    "http://127.0.0.1:8000/api/shelf-event"
)

CONFIDENCE_THRESHOLD = 0.45

COOLDOWN_FRAMES = 60


# =========================================================
# SHELF REGIONS
#
# Change these according to your actual camera.
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
# =========================================================

PRODUCTS = {

    0: {

        "name":
            "Potato Chips",

        "barcode":
            "890100000001",

        "expected_shelf":
            "SHELF_A"

    },


    1: {

        "name":
            "Biscuits",

        "barcode":
            "890100000002",

        "expected_shelf":
            "SHELF_A"

    },


    2: {

        "name":
            "Orange Juice",

        "barcode":
            "890100000003",

        "expected_shelf":
            "SHELF_B"

    },


    3: {

        "name":
            "Bath Soap",

        "barcode":
            "890100000004",

        "expected_shelf":
            "SHELF_C"

    },


    4: {

        "name":
            "Milk Pack",

        "barcode":
            "890100000005",

        "expected_shelf":
            "SHELF_B"

    },


    5: {

        "name":
            "Cola Can",

        "barcode":
            "890100000006",

        "expected_shelf":
            "SHELF_B"

    }

}


# =========================================================
# FIND SHELF
# =========================================================

def find_shelf(
    x,
    y
):

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


    except requests.RequestException as error:

        print(
            "Shelf API error:",
            error
        )


# =========================================================
# MODEL
# =========================================================

model = YOLO(
    MODEL_PATH
)


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


cooldowns = defaultdict(int)

frame_number = 0


try:

    while True:

        success, frame = cap.read()


        if not success:

            break


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


        # -------------------------------------------------
        # YOLO PRODUCT DETECTION
        # -------------------------------------------------

        results = model(
            frame,
            verbose=False
        )


        for result in results:

            if result.boxes is None:

                continue


            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )


                if (
                    confidence
                    <
                    CONFIDENCE_THRESHOLD
                ):

                    continue


                class_id = int(
                    box.cls[0]
                )


                product = PRODUCTS.get(
                    class_id
                )


                if product is None:

                    continue


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


                detected_shelf = find_shelf(

                    center_x,

                    center_y

                )


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


                if status == "MISPLACED":

                    box_color = (
                        0,
                        0,
                        255
                    )

                else:

                    box_color = (
                        0,
                        255,
                        0
                    )


                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    box_color,

                    2

                )


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
                # Send only occasionally to avoid duplicate alerts
                # -------------------------------------------------

                if (

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


        cv2.imshow(

            "RetailEdge AI - Shelf AI",

            frame

        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q") or key == 27:

            break


        frame_number += 1


finally:

    cap.release()

    cv2.destroyAllWindows()