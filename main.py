import cv2
import mediapipe as mp
import numpy as np

# drawings for our body pose
mp_drawing = mp.solutions.drawing_utils
# importing pose estimation models from mediapipe
mp_pose = mp.solutions.pose

# VIDEO FEED
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('Mediapipe Feed', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
