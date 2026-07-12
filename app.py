from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import os
import winsound
import requests
import time

app = Flask(__name__)

# -------------------------------
# PUSHOVER CONFIG
# -------------------------------
USER_KEY = "u1kudxtf3xw949zc6zm3d89es7yfsp"
API_TOKEN = "axtmi7fu6pbjwyan3qnzu7ee2jxww7"


def send_notification(message):
    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": API_TOKEN,
                "user": USER_KEY,
                "message": message
            },
            timeout=5
        )
        print("Notification sent")
    except Exception as e:
        print("Notification error:", e)


# -------------------------------
# LOAD YOLO MODEL
# -------------------------------
model = YOLO("yolov8x.pt")

# -------------------------------
# INPUT SOURCE
# -------------------------------
source = "image1.png"

image_extensions = [".jpg", ".jpeg", ".png"]
is_image = os.path.splitext(source)[1].lower() in image_extensions

# -------------------------------
# GLOBAL VARIABLES
# -------------------------------
people_count = 0
density_threshold = 30
last_alert_time = 0
alert_cooldown = 30

# For unique tracking
unique_ids = set()

# -------------------------------
# VIDEO CAPTURE
# -------------------------------
if not is_image:
    cap = cv2.VideoCapture(source)


# -------------------------------
# FRAME GENERATOR
# -------------------------------
def generate_frames():
    global people_count
    global density_threshold
    global last_alert_time
    global unique_ids

    # ============================
    # IMAGE MODE
    # ============================
    if is_image:

        image = cv2.imread(source)

        results = model(
            source,
            conf=0.25,
            iou=0.35,
            imgsz=1280
        )

        person_count = 0

        for result in results:
            for box in result.boxes:

                if int(box.cls[0]) == 0:

                    person_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

        people_count = person_count

        cv2.putText(
            image,
            f"Total Persons: {person_count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        if person_count > density_threshold:

            cv2.putText(
                image,
                "ALERT: High Crowd Density!",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            current_time = time.time()

            if current_time - last_alert_time > alert_cooldown:

                winsound.Beep(2000, 600)

                send_notification(
                    f"Crowd alert! {person_count} people detected."
                )

                last_alert_time = current_time

        ret, buffer = cv2.imencode(".jpg", image)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame +
            b"\r\n"
        )

    # ============================
    # VIDEO MODE
    # ============================
    else:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Resize for stable detection
            frame = cv2.resize(frame, (1280, 720))

            results = model.track(
                frame,
                conf=0.3,
                iou=0.4,
                imgsz=1280,
                persist=True
            )

            if results[0].boxes.id is not None:

                for box, track_id, cls in zip(
                    results[0].boxes.xyxy,
                    results[0].boxes.id,
                    results[0].boxes.cls
                ):

                    if int(cls) == 0:

                        unique_ids.add(int(track_id))

                        x1, y1, x2, y2 = map(int, box)

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

            total_people = len(unique_ids)

            people_count = total_people

            cv2.putText(
                frame,
                f"Unique Persons: {total_people}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            if total_people > density_threshold:

                cv2.putText(
                    frame,
                    "ALERT: High Crowd Density!",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                current_time = time.time()

                if current_time - last_alert_time > alert_cooldown:

                    winsound.Beep(2000, 600)

                    send_notification(
                        f"Crowd alert! {total_people} people detected."
                    )

                    last_alert_time = current_time

            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )


# -------------------------------
# ROUTES
# -------------------------------
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/count")
def count():
    return jsonify({"count": people_count})


@app.route("/set_threshold", methods=["POST"])
def set_threshold():
    global density_threshold
    density_threshold = int(request.form["value"])
    return "OK"


# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
