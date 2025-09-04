import cv2
import numpy as np
import mediapipe as mp
from modules.posture.utils.angle_utils import calculate_angle

mp_pose = mp.solutions.pose


class CurlCounter:
    def __init__(self):
        # Separate counters for both arms
        self.left_counter = 0
        self.left_stage = None

        self.right_counter = 0
        self.right_stage = None

    def process(self, image, landmarks):
        # ---- LEFT ARM ----
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        cv2.putText(image, str(int(left_angle)),
                    tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        if left_angle > 160:
            self.left_stage = "down"
        if left_angle < 30 and self.left_stage == "down":
            self.left_stage = "up"
            self.left_counter += 1
            print(f"Left Reps: {self.left_counter}")

        # ---- RIGHT ARM ----
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        cv2.putText(image, str(int(right_angle)),
                    tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        if right_angle > 160:
            self.right_stage = "down"
        if right_angle < 30 and self.right_stage == "down":
            self.right_stage = "up"
            self.right_counter += 1
            print(f"Right Reps: {self.right_counter}")

        # ---- DRAW COUNTERS ON SCREEN ----
        cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)

        # Left reps
        cv2.putText(image, 'LEFT REPS', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f"{self.left_counter} ({self.left_stage if self.left_stage else '-'})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Right reps
        cv2.putText(image, 'RIGHT REPS', (150, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f"{self.right_counter} ({self.right_stage if self.right_stage else '-'})",
                    (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
