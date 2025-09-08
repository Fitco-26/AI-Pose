import cv2
import numpy as np
import mediapipe as mp
import time
from .posture.utils.angle_utils import calculate_angle

mp_pose = mp.solutions.pose


class CurlCounter:
    def __init__(self):
        # Separate counters for both arms
        self.left_counter = 0
        self.left_stage = None
        self.left_feedback = ""
        self.left_form_fault = False  # Flag to track form during a single rep
        self.left_angle = 160.0
        self.left_rep_time = 0

        self.right_counter = 0
        self.right_stage = None
        self.right_feedback = ""
        self.right_form_fault = False  # Flag to track form during a single rep
        self.right_angle = 160.0
        self.right_rep_time = 0

    def _process_arm(self, shoulder, elbow, wrist, hip, arm_side: str):
        """Helper function to process a single arm's logic."""
        # Get current state for the specified arm
        counter = self.left_counter if arm_side == 'left' else self.right_counter
        stage = self.left_stage if arm_side == 'left' else self.right_stage
        form_fault = self.left_form_fault if arm_side == 'left' else self.right_form_fault
        elbow_angle = self.left_angle if arm_side == 'left' else self.right_angle
        rep_time = self.left_rep_time if arm_side == 'left' else self.right_rep_time
        feedback = ""

        # --- Angle & Form Calculations ---
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        shoulder_angle = calculate_angle(hip, shoulder, elbow)

        # --- Multi-Point Form Fault Logic ---
        # 1. Arm Swinging
        if shoulder_angle > 45:
            feedback = f"{arm_side[0].upper()}: Keep Elbow Still"
            if stage == 'down':
                form_fault = True

        # 2. Elbow Flaring
        if elbow_angle > 150 and shoulder_angle > 80:
            feedback = f"{arm_side[0].upper()}: Keep Arm Down"
            if stage == 'down':
                form_fault = True

        # --- State Machine for Rep Counting ---
        if elbow_angle > 160:  # Arm is in the "down" position
            if stage == 'up':  # Just completed a rep
                form_fault = False  # Reset fault for the next rep
            stage = "down"
        elif elbow_angle < 30 and stage == "down":  # Arm is in the "up" position
            # 3. Tempo Check: Rep is too fast
            if rep_time > 0 and time.time() - rep_time < 0.8:  # Less than 0.8s for the 'up' phase
                feedback = "Too Fast!"
                form_fault = True

            stage = "up"
            rep_time = time.time()  # Start timer for the 'up' phase

            if not form_fault:
                counter += 1
                print(f"{arm_side.capitalize()} Reps: {counter}")

        # If a fault was triggered, but the user corrects mid-rep, clear the message.
        # But the rep itself remains invalid until reset.
        if not feedback and not form_fault:
            feedback = ""

        # --- Update instance attributes ---
        if arm_side == 'left':
            self.left_counter, self.left_stage, self.left_form_fault, self.left_feedback, self.left_angle, self.left_rep_time = counter, stage, form_fault, feedback, elbow_angle, rep_time
        else:
            self.right_counter, self.right_stage, self.right_form_fault, self.right_feedback, self.right_angle, self.right_rep_time = counter, stage, form_fault, feedback, elbow_angle, rep_time

    def process(self, image, landmarks):
        try:
            # Extract landmark coordinates
            lm = landmarks
            lm_pose = mp_pose.PoseLandmark

            # Left Arm
            left_shoulder = [lm[lm_pose.LEFT_SHOULDER.value].x,
                             lm[lm_pose.LEFT_SHOULDER.value].y]
            left_elbow = [lm[lm_pose.LEFT_ELBOW.value].x,
                          lm[lm_pose.LEFT_ELBOW.value].y]
            left_wrist = [lm[lm_pose.LEFT_WRIST.value].x,
                          lm[lm_pose.LEFT_WRIST.value].y]
            left_hip = [lm[lm_pose.LEFT_HIP.value].x,
                        lm[lm_pose.LEFT_HIP.value].y]
            self._process_arm(left_shoulder, left_elbow,
                              left_wrist, left_hip, 'left')

            # Right Arm
            right_shoulder = [lm[lm_pose.RIGHT_SHOULDER.value].x,
                              lm[lm_pose.RIGHT_SHOULDER.value].y]
            right_elbow = [lm[lm_pose.RIGHT_ELBOW.value].x,
                           lm[lm_pose.RIGHT_ELBOW.value].y]
            right_wrist = [lm[lm_pose.RIGHT_WRIST.value].x,
                           lm[lm_pose.RIGHT_WRIST.value].y]
            right_hip = [lm[lm_pose.RIGHT_HIP.value].x,
                         lm[lm_pose.RIGHT_HIP.value].y]
            self._process_arm(right_shoulder, right_elbow,
                              right_wrist, right_hip, 'right')
        except (IndexError, KeyError):
            # Handle cases where some landmarks are not visible
            print("Warning: Not all required landmarks are visible.")
