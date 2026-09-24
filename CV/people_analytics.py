import cv2
import time
import requests
from ultralytics import YOLO

# =========================================================
# SETTINGS
# =========================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR /  "yolo11n.pt"
VIDEO_SOURCE = 0

API_URL = "http://127.0.0.1:8000/api/shopper"

model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(VIDEO_SOURCE)

# =========================================================
# ZONES
# Change these coordinates according to your camera view
# =========================================================

ZONES = {

    # Left half — one large zone
    "zone_a": (0, 0, 320, 480),

    # Right half — divided into two
    "zone_b": (320, 0, 640, 240),

    "zone_c": (320, 240, 640, 480)

}
# =========================================================
# DWELL TIME DATA
# =========================================================

# Stores when a person entered each zone
zone_entry_times = {
    "zone_a": {},
    "zone_b": {},
    "zone_c": {}
}

# Stores completed dwell times
completed_dwell_times = {
    "zone_a": [],
    "zone_b": [],
    "zone_c": []
}

# Send data to backend every few seconds
last_api_send = 0
API_INTERVAL = 2


# =========================================================
# CHECK WHETHER POINT IS INSIDE ZONE
# =========================================================

def point_inside_zone(x, y, zone):

    x1, y1, x2, y2 = zone

    return x1 <= x <= x2 and y1 <= y <= y2


# =========================================================
# FIND WHICH ZONE THE PERSON IS IN
# =========================================================

def get_person_zone(x, y):

    for zone_name, zone_coordinates in ZONES.items():

        if point_inside_zone(x, y, zone_coordinates):
            return zone_name

    return None


# =========================================================
# SEND DATA TO BACKEND
# =========================================================

def send_to_backend(customers_inside, zone_counts, average_dwell):

    data = {
        "customers_inside": customers_inside,
        "zone_a": zone_counts["zone_a"],
        "zone_b": zone_counts["zone_b"],
        "zone_c": zone_counts["zone_c"],
        "average_dwell_seconds": average_dwell
    }

    try:

        response = requests.post(
            API_URL,
            json=data,
            timeout=2
        )

        print("Backend:", response.json())

    except Exception as e:

        print("Backend connection error:", e)


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera/video stopped.")
        break

    # -----------------------------------------------------
    # YOLO TRACKING
    # -----------------------------------------------------

    results = model.track(
        frame,
        conf=0.65,
        persist=True,
        classes=[0],      # class 0 = person
        verbose=False
    )

    current_time = time.time()

    # Current people in each zone
    current_zone_people = {
        "zone_a": set(),
        "zone_b": set(),
        "zone_c": set()
    }

    # -----------------------------------------------------
    # PROCESS DETECTIONS
    # -----------------------------------------------------

    for result in results:

        if result.boxes.id is None:
            continue

        boxes = result.boxes

        track_ids = boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes.xyxy, track_ids):

            x1, y1, x2, y2 = map(int, box)

            # Center of person
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Find person's zone
            zone = get_person_zone(center_x, center_y)

            # -------------------------------------------------
            # DRAW PERSON
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # -------------------------------------------------
            # DWELL TIME
            # -------------------------------------------------

            if zone is not None:

                current_zone_people[zone].add(track_id)

                # Person entered this zone for first time
                if track_id not in zone_entry_times[zone]:

                    zone_entry_times[zone][track_id] = current_time

                # Calculate current dwell time
                dwell_seconds = (
                    current_time -
                    zone_entry_times[zone][track_id]
                )

                # Display zone + dwell time
                cv2.putText(
                    frame,
                    f"{zone} {dwell_seconds:.1f}s",
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

    # =====================================================
    # PROCESS PEOPLE WHO LEFT A ZONE
    # =====================================================

    for zone in ZONES:

        tracked_people = set(
            zone_entry_times[zone].keys()
        )

        for track_id in list(tracked_people):

            if track_id not in current_zone_people[zone]:

                # Person has left the zone
                dwell_seconds = (
                    current_time -
                    zone_entry_times[zone][track_id]
                )

                completed_dwell_times[zone].append(
                    dwell_seconds
                )

                print(
                    f"Person {track_id} left {zone} "
                    f"after {dwell_seconds:.1f} seconds"
                )

                # Remove person from active tracking
                del zone_entry_times[zone][track_id]

    # =====================================================
    # ZONE COUNTS
    # =====================================================

    zone_counts = {
        "zone_a": len(current_zone_people["zone_a"]),
        "zone_b": len(current_zone_people["zone_b"]),
        "zone_c": len(current_zone_people["zone_c"])
    }

    customers_inside = sum(zone_counts.values())

    # =====================================================
    # CALCULATE AVERAGE DWELL TIME
    # =====================================================

    all_completed_times = []

    for zone in completed_dwell_times:

        all_completed_times.extend(
            completed_dwell_times[zone]
        )

    if len(all_completed_times) > 0:

        average_dwell = (
            sum(all_completed_times) /
            len(all_completed_times)
        )

    else:

        average_dwell = 0

    # =====================================================
    # DISPLAY INFORMATION
    # =====================================================

    cv2.putText(
        frame,
        f"Customers: {customers_inside}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Average Dwell: {average_dwell:.1f}s",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # =====================================================
    # DRAW ZONES
    # =====================================================

    for zone_name, (x1, y1, x2, y2) in ZONES.items():

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            zone_name.upper(),
            (x1 + 10, y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    # =====================================================
    # SEND TO BACKEND
    # =====================================================

    if current_time - last_api_send >= API_INTERVAL:

        send_to_backend(
            customers_inside,
            zone_counts,
            average_dwell
        )

        last_api_send = current_time

    # =====================================================
    # SHOW CAMERA
    # =====================================================

    cv2.imshow(
        "RetailEdge AI - Shopper Analytics",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()