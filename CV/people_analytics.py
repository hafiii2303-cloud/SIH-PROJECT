import cv2
import requests
from ultralytics import YOLO


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "yolo11n.pt"

CAMERA_INDEX = 0

API_URL = "http://127.0.0.1:8000/api/shopper"

CONFIDENCE_THRESHOLD = 0.40

SEND_EVERY_N_FRAMES = 10


# =========================================================
# ZONES
# Change these according to your camera view
# x1, y1, x2, y2
# =========================================================

ZONES = {

    "zone_a": (0, 0, 426, 240),

    "zone_b": (426, 0, 852, 240),

    "zone_c": (0, 240, 852, 480)

}


# =========================================================
# FUNCTIONS
# =========================================================

def point_inside(
    x,
    y,
    box
):

    x1, y1, x2, y2 = box

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )


def send_to_backend(
    people,
    zone_counts
):

    data = {

        "customers_inside":
            people,

        "zone_a":
            zone_counts["zone_a"],

        "zone_b":
            zone_counts["zone_b"],

        "zone_c":
            zone_counts["zone_c"]

    }

    try:

        response = requests.post(

            API_URL,

            json=data,

            timeout=1.5

        )

        response.raise_for_status()

    except requests.RequestException as error:

        print(
            "Shopper API error:",
            error
        )


# =========================================================
# LOAD MODEL
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
        "Camera could not be opened. "
        "Try CAMERA_INDEX = 1."
    )


# =========================================================
# MAIN LOOP
# =========================================================

frame_number = 0


try:

    while True:

        success, frame = cap.read()


        if not success:

            print(
                "Could not read camera frame."
            )

            break


        frame = cv2.resize(
            frame,
            (852, 480)
        )


        zone_counts = {

            "zone_a": 0,

            "zone_b": 0,

            "zone_c": 0

        }


        people_count = 0


        # -------------------------------------------------
        # YOLO TRACKING
        # -------------------------------------------------

        results = model.track(

            frame,

            persist=True,

            classes=[0],  # person

            verbose=False

        )


        if results:

            result = results[0]


            if result.boxes is not None:

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


                    people_count += 1


                    # -------------------------------------
                    # Zone detection
                    # -------------------------------------

                    for zone_name, zone_box in ZONES.items():

                        if point_inside(
                            center_x,
                            center_y,
                            zone_box
                        ):

                            zone_counts[
                                zone_name
                            ] += 1

                            break


                    # -------------------------------------
                    # Draw person
                    # -------------------------------------

                    cv2.rectangle(

                        frame,

                        (x1, y1),

                        (x2, y2),

                        (0, 255, 0),

                        2

                    )


                    label = "Person"


                    if box.id is not None:

                        label += (
                            f" ID:"
                            f"{int(box.id[0])}"
                        )


                    label += (
                        f" "
                        f"{confidence:.2f}"
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

                        0.55,

                        (0, 255, 0),

                        2

                    )


        # -------------------------------------------------
        # Draw zones
        # -------------------------------------------------

        for zone_name, (
            x1,
            y1,
            x2,
            y2
        ) in ZONES.items():


            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (255, 180, 0),

                2

            )


            cv2.putText(

                frame,

                (
                    f"{zone_name}: "
                    f"{zone_counts[zone_name]}"
                ),

                (
                    x1 + 8,
                    y1 + 25
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.60,

                (255, 180, 0),

                2

            )


        # -------------------------------------------------
        # Display occupancy
        # -------------------------------------------------

        cv2.putText(

            frame,

            f"Customers: {people_count}",

            (15, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.85,

            (0, 255, 255),

            2

        )


        # -------------------------------------------------
        # Send to FastAPI
        # -------------------------------------------------

        if (
            frame_number
            %
            SEND_EVERY_N_FRAMES
            ==
            0
        ):

            send_to_backend(
                people_count,
                zone_counts
            )


        cv2.imshow(

            "RetailEdge AI - Shopper Analytics",

            frame

        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q") or key == 27:

            break


        frame_number += 1


finally:

    cap.release()

    cv2.destroyAllWindows()