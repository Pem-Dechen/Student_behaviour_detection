
from ultralytics import YOLO
import cv2
import pyttsx3
import time
import os
from datetime import datetime

# ==========================================
# LOAD MODEL
# ==========================================
model = YOLO(
    r"C:\Users\dell\runs\detect\train-6\weights\best.pt"
)

# ==========================================
# TEXT TO SPEECH
# ==========================================
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# ==========================================
# SCREENSHOT FOLDER
# ==========================================
SCREENSHOT_FOLDER = "Screenshots"
os.makedirs(
    SCREENSHOT_FOLDER,
    exist_ok=True
)

# ==========================================
# SAVE SCREENSHOT
# ==========================================
def save_screenshot(frame, behavior):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"{behavior}_{timestamp}.jpg"
    )

    filepath = os.path.join(
        SCREENSHOT_FOLDER,
        filename
    )

    cv2.imwrite(filepath, frame)

    print(
        f"Screenshot Saved: {filepath}"
    )

# ==========================================
# ALERT SETTINGS
# ==========================================
last_alert = {}

ALERT_COOLDOWN = 5

STATUS_COOLDOWN = 15

last_status_voice = 0

# ==========================================
# CAMERA
# ==========================================
cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

if not cap.isOpened():

    print("Cannot open webcam")

    exit()

cv2.namedWindow(
    "Student Engagement Detection System",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Student Engagement Detection System",
    1400,
    900
)

print("=" * 50)
print("Student Engagement Detection System")
print("Press Q to Exit")
print("=" * 50)

# ==========================================
# MAIN LOOP
# ==========================================
while True:

    success, frame = cap.read()

    if not success:
        break

    current_time = time.time()

    results = model(
        frame,
        conf=0.60,
        verbose=False
    )

    # ======================================
    # COUNTERS
    # ======================================
    study_count = 0
    phone_count = 0
    sleep_count = 0

    highest_confidence = 0

    current_activity = (
        "No Student Detected"
    )

    # ======================================
    # DETECTIONS
    # ======================================
    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        confidence = float(
            box.conf[0]
        )

        class_name = model.names[
            cls_id
        ].lower()

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # ==============================
        # STUDY
        # ==============================
        if class_name == "study":

            study_count += 1

            color = (0, 255, 0)

            label = (
                f"Studying "
                f"{confidence:.2f}"
            )

            if (
                confidence >
                highest_confidence
            ):

                highest_confidence = (
                    confidence
                )

                current_activity = (
                    "Studying"
                )

        # ==============================
        # PHONE
        # ==============================
        elif class_name == "phone":

            phone_count += 1

            color = (
                0,
                165,
                255
            )

            label = (
                f"Using Phone "
                f"{confidence:.2f}"
            )

            if (
                confidence >
                highest_confidence
            ):

                highest_confidence = (
                    confidence
                )

                current_activity = (
                    "Using Phone"
                )

            if (
                current_time -
                last_alert.get(
                    "phone",
                    0
                )
                > ALERT_COOLDOWN
            ):

                save_screenshot(
                    frame,
                    "phone"
                )

                engine.say(
                    "Warning. Student is using a mobile phone."
                )

                engine.runAndWait()

                last_alert[
                    "phone"
                ] = current_time

        # ==============================
        # SLEEP
        # ==============================
        elif class_name == "sleep":

            sleep_count += 1

            color = (
                0,
                0,
                255
            )

            label = (
                f"Sleeping "
                f"{confidence:.2f}"
            )

            if (
                confidence >
                highest_confidence
            ):

                highest_confidence = (
                    confidence
                )

                current_activity = (
                    "Sleeping"
                )

            if (
                current_time -
                last_alert.get(
                    "sleep",
                    0
                )
                > ALERT_COOLDOWN
            ):

                save_screenshot(
                    frame,
                    "sleep"
                )

                engine.say(
                    "Warning. Student is sleeping."
                )

                engine.runAndWait()

                last_alert[
                    "sleep"
                ] = current_time

        else:

            color = (
                255,
                255,
                0
            )

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

        # ==============================
        # DRAW BOX
        # ==============================
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # ======================================
    # ATTENTION SCORE
    # ======================================
    total = (
        study_count +
        phone_count +
        sleep_count
    )

    if total > 0:

        attention_percentage = (
            study_count / total
        ) * 100

        attention_text = (
            f"{attention_percentage:.1f}%"
        )

    else:

        attention_percentage = 0

        attention_text = "N/A"

    # ======================================
    # CONCLUSION
    # ======================================
    if total == 0:

        conclusion = (
            "Waiting for Student"
        )

        conclusion_color = (
            255,
            255,
            255
        )

    elif study_count > (
        phone_count +
        sleep_count
    ):

        conclusion = (
            "Student is Engaged"
        )

        conclusion_color = (
            0,
            255,
            0
        )

    else:

        conclusion = (
            "Student is Distracted"
        )

        conclusion_color = (
            0,
            0,
            255
        )

    # ======================================
    # VOICE CONCLUSION
    # ======================================
    if (
        current_time -
        last_status_voice
        > STATUS_COOLDOWN
    ):

        if (
            conclusion ==
            "Student is Engaged"
        ):

            engine.say(
                "Student is engaged."
            )

        elif (
            conclusion ==
            "Student is Distracted"
        ):

            engine.say(
                "Student is distracted."
            )

        engine.runAndWait()

        last_status_voice = (
            current_time
        )

    # ======================================
    # DASHBOARD
    # ======================================
    cv2.rectangle(
        frame,
        (0, 0),
        (700, 180),
        (40, 40, 40),
        -1
    )

    cv2.putText(
        frame,
        "STUDENT ENGAGEMENT SYSTEM",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Activity: {current_activity}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Attention: {attention_text}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Conclusion: {conclusion}",
        (10, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        conclusion_color,
        2
    )

    # ======================================
    # DISPLAY
    # ======================================
    cv2.imshow(
        "Student Engagement Detection System",
        frame
    )

    # PRESS Q TO EXIT
    if (
        cv2.waitKey(1)
        & 0xFF
        == ord("q")
    ):
        break

# ==========================================
# CLOSE
# ==========================================
cap.release()

cv2.destroyAllWindows()

print("\nSystem Closed Successfully")

