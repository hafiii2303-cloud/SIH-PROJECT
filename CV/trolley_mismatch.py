import cv2
import requests
from ultralytics import YOLO


MODEL_PATH = "models/retail_products.pt"

CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.50

API_URL = (
    "http://127.0.0.1:8000/api/product-mismatch"
)


PRODUCTS = {

    0: {

        "name":
            "Potato Chips",

        "barcode":
            "890100000001"

    },

    1: {

        "name":
            "Biscuits",

        "barcode":
            "890100000002"

    },

    2: {

        "name":
            "Orange Juice",

        "barcode":
            "890100000003"

    },

    3: {

        "name":
            "Bath Soap",

        "barcode":
            "890100000004"

    },

    4: {

        "name":
            "Milk Pack",

        "barcode":
            "890100000005"

    },

    5: {

        "name":
            "Cola Can",

        "barcode":
            "890100000006"

    }

}


def send_mismatch(
    trolley_id,
    expected_barcode,
    detected_barcode,
    confidence
):

    data = {

        "trolley_id":
            trolley_id,

        "expected_barcode":
            expected_barcode,

        "detected_barcode":
            detected_barcode,

        "confidence":
            confidence

    }


    try:

        response = requests.post(

            API_URL,

            json=data,

            timeout=1.5

        )

        response.raise_for_status()

        print(
            "Mismatch sent:",
            response.json()
        )


    except requests.RequestException as error:

        print(
            "Mismatch API error:",
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
        "Could not open trolley camera."
    )


# For this first version enter the
# scanned barcode manually.
expected_barcode = input(
    "Enter scanned product barcode: "
).strip()


last_alert_frame = -9999

ALERT_COOLDOWN = 90

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


        best = None


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


                if (
                    best is None
                    or
                    confidence
                    >
                    best["confidence"]
                ):

                    best = {

                        "name":
                            product["name"],

                        "barcode":
                            product["barcode"],

                        "confidence":
                            confidence,

                        "box":
                            box

                    }


        status = (
            "WAITING FOR PRODUCT"
        )


        if best is not None:

            detected_barcode = (
                best["barcode"]
            )


            if (
                detected_barcode
                ==
                expected_barcode
            ):

                status = "MATCH"

                color = (
                    0,
                    255,
                    0
                )

            else:

                status = "MISMATCH"

                color = (
                    0,
                    0,
                    255
                )


                if (
                    frame_number
                    >=
                    last_alert_frame
                    +
                    ALERT_COOLDOWN
                ):

                    send_mismatch(

                        "TROLLEY_01",

                        expected_barcode,

                        detected_barcode,

                        best["confidence"]

                    )


                    last_alert_frame = (
                        frame_number
                    )


            box = best["box"]


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                color,

                2

            )


            cv2.putText(

                frame,

                (
                    f"{best['name']} "
                    f"{best['confidence']:.2f}"
                ),

                (
                    x1,
                    max(
                        20,
                        y1 - 8
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.60,

                color,

                2

            )


        cv2.putText(

            frame,

            (
                f"Expected: "
                f"{expected_barcode}"
            ),

            (15, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 0),

            2

        )


        cv2.putText(

            frame,

            f"Status: {status}",

            (15, 65),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (
                0, 255, 0
            )
            if status == "MATCH"
            else (
                0, 0, 255
            )
            if status == "MISMATCH"
            else (
                255, 255, 0
            ),

            2

        )


        cv2.imshow(

            "RetailEdge AI - Trolley Verification",

            frame

        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q") or key == 27:

            break


        frame_number += 1


finally:

    cap.release()

    cv2.destroyAllWindows()