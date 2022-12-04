import time
import cv2
from hand_tracking import HandDetector

# Camera capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# For FPS measurement
prev_time = time.perf_counter()

hand_detector = HandDetector()

while cap.isOpened():
    success, img = cap.read()

    img, landmarks = hand_detector.find_hands(img, return_landmarks=True)

    # Print landmarks to console
    if len(landmarks) != 0:
        print(landmarks[0][0])

    # FPS measurement and showing on screen
    curr_time = time.perf_counter()
    fps = 1 / (curr_time - prev_time)
    prev_time = time.perf_counter()
    cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, color=(255, 0, 0), thickness=3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
