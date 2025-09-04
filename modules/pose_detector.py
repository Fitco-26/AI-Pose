import cv2
import mediapipe as mp

from modules.posture.utils.drawing_utils import (
    mp_drawing,
    mp_pose,
    landmark_style,
    connection_style
)


class PoseDetector:
    def __init__(self, detection_conf=0.5, tracking_conf=0.5):
        self.pose = mp_pose.Pose(
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )

    def detect(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_landmarks(self, image, results):
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_style,
                connection_style
            )
