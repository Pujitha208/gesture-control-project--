print("START OF PROGRAM")

import cv2
import sys
import time
import pyautogui
import importlib

# Import MediaPipe safely
try:
    mp = importlib.import_module("mediapipe")
except ImportError:
    print("Error: mediapipe is not installed.")
    print("Run: pip install mediapipe")
    sys.exit(1)

# Screen size
screen_w, screen_h = pyautogui.size()

# Open webcam (Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Give camera time to start
time.sleep(2)

if not cap.isOpened():
    print("Error: Camera could not be opened.")
    input("Press Enter to exit...")
    sys.exit()

print("Camera opened successfully.")

# MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

draw = mp.solutions.drawing_utils

while True:
    success, frame = cap.read()

    if not success:
        continue

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness):

            draw.draw_landmarks(
                frame,
                hand_landmarks,
                mpHands.HAND_CONNECTIONS
            )

            label = handedness.classification[0].label

            # Right hand controls mouse
            if label == "Right":

                x = hand_landmarks.landmark[8].x
                y = hand_landmarks.landmark[8].y

                mouse_x = int(x * screen_w)
                mouse_y = int(y * screen_h)

                pyautogui.moveTo(mouse_x, mouse_y)

                cv2.circle(
                    frame,
                    (int(x * frame.shape[1]), int(y * frame.shape[0])),
                    10,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    "Mouse Control",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Gesture Mouse", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()