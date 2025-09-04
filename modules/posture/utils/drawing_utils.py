import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

landmark_style = mp_drawing.DrawingSpec(
    color=(0, 255, 0), thickness=2, circle_radius=2)
connection_style = mp_drawing.DrawingSpec(
    color=(255, 0, 0), thickness=2, circle_radius=2)
