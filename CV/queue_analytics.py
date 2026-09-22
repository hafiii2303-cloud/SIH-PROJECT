import cv2
import requests
from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"

CAMERA_INDEX = 0

API_URL = "http://127.0.0.1:8000/api/queue"

CONFIDENCE_THRESHOLD = 0.40

ACTIVE_COUNTERS = 1

AVERAGE_SERVICE_SECONDS = 60

SEND_EVERY_N_FRAMES = 10


# x1, y1, x2, y2
QUEUE_ROI = (
    120,
    100,
    700,
    440
)


def inside_queue(
    x,
    y
):

    x1, y1, x2, y2 = QUEUE_ROI

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )


def send_queue(
    people
):

    data = {

        "people":
            people,

        "active_counters":
            ACTIVE_COUNTERS,

        "average_service_seconds":
            AVERAGE_SERVICE_SECONDS

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
            "Queue API error:",
            error
        )


model = YOLO(
    MODEL_PATH
)


cap = cv2.VideoCapture(
    CAMERA_INDEX
)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open camera."
    )


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


        queue_count = 0


        results = model.track(

            frame,

            persist=True,

            classes=[0],

            verbose=False

        )


        if results:

            result = results[0]


            if result.boxes is not None:

                for box in result.boxes:

                    confidence = float(
                        box.conf[0]
                    )


                    if confidence < CONFIDENCE_THRESHOLD:

                        continue


                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )


                    cx = int(
                        (x1 + x2) / 2
                    )


                    cy = int(
                        (y1 + y2) / 2
                    )


                    if inside_queue(
                        cx,
                        cy
                    ):

                        queue_count += 1


                        cv2.rectangle(

                            frame,

                            (x1, y1),

                            (x2, y2),

                            (0, 255, 0),

                            2

                        )


        # -------------------------------------------------
        # Draw queue ROI
        # -------------------------------------------------

        x1, y1, x2, y2 = QUEUE_ROI


        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            (0, 0, 255),

            2

        )


        # -------------------------------------------------
        # Estimate wait
        # -------------------------------------------------

        wait_minutes = (

            queue_count

            *

            AVERAGE_SERVICE_SECONDS

            /

            max(
                1,
                ACTIVE_COUNTERS
            )

            /

            60

        )


        cv2.putText(

            frame,

            f"Queue: {queue_count}",

            (15, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.80,

            (0, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Wait: {wait_minutes:.1f} min",

            (15, 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.70,

            (0, 255, 255),

            2

        )


        if (
            frame_number
            %
            SEND_EVERY_N_FRAMES
            ==
            0
        ):

            send_queue(
                queue_count
            )


        cv2.imshow(

            "RetailEdge AI - Queue",

            frame

        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q") or key == 27:

            break


        frame_number += 1


finally:

    cap.release()

    cv2.destroyAllWindows()