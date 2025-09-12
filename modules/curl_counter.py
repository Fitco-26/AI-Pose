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
        self.left_stage = 'down'
        self.left_feedback = ""
        self.left_form_fault = False
        self.left_angle = 160.0
        self.left_rep_time = 0
        self.left_min_angle = 180
        self.left_max_angle = 0

        self.right_counter = 0
        self.right_stage = 'down'
        self.right_feedback = ""
        self.right_form_fault = False
        self.right_angle = 160.0
        self.right_rep_time = 0
        self.right_min_angle = 180
        self.right_max_angle = 0

        # Error logging
        self.new_error_logged = False
        self.last_error = (None, None)

    def _process_arm(self, shoulder, elbow, wrist, hip, index_finger, arm_side: str, landmarks):
        """Helper function to process a single arm's logic."""
        counter = self.left_counter if arm_side == 'left' else self.right_counter
        stage = self.left_stage if arm_side == 'left' else self.right_stage
        form_fault = self.left_form_fault if arm_side == 'left' else self.right_form_fault
        elbow_angle = self.left_angle if arm_side == 'left' else self.right_angle
        rep_time = self.left_rep_time if arm_side == 'left' else self.right_rep_time
        min_angle = self.left_min_angle if arm_side == 'left' else self.right_min_angle
        max_angle = self.left_max_angle if arm_side == 'left' else self.right_max_angle
        feedback = ""

        # Extract full landmark data for Z-coordinate
        lm_pose = mp_pose.PoseLandmark
        shoulder_lm = landmarks[lm_pose[f'{arm_side.upper()}_SHOULDER'].value]
        wrist_lm = landmarks[lm_pose[f'{arm_side.upper()}_WRIST'].value]

        # --- Angle & Form Calculations ---
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        shoulder_angle = calculate_angle(hip, shoulder, elbow)
        wrist_angle = calculate_angle(elbow, wrist, index_finger)
        elbow_shoulder_dist_x = abs(elbow[0] - shoulder[0])
        torso_angle = calculate_angle(
            [hip[0], hip[1] + 0.5], hip, shoulder)  # Angle relative to vertical

        # --- Prioritized Form Fault Logic ---
        # Reset form fault flag at the start of each frame check
        form_fault = False

        # 1. Torso Leaning / Using Momentum
        if not (165 < torso_angle < 195):
            feedback = "Don’t Lean Back"
            form_fault = True
        # 2. Arm Swinging (Shoulder moving too much)
        elif shoulder_angle > 35:
            feedback = "Keep Elbow Still"
            form_fault = True
        # 3. Elbow Too High (lifting shoulder instead of curling)
        elif elbow[1] < shoulder[1]:  # elbow above shoulder in y-axis
            feedback = "Keep Elbow Down"
            form_fault = True
        # 4. Hyperextension
        elif elbow_angle > 190:
            feedback = "Don’t Overextend"
            form_fault = True
        # 5. Elbows Flaring Outward
        elif elbow_shoulder_dist_x > 0.1:  # Heuristic value, may need tuning
            feedback = "Keep Elbows Tucked In"
            form_fault = True
        # 6. Wrist Bending (check during contraction)
        elif elbow_angle < 120 and not (150 < wrist_angle < 210):
            feedback = "Keep Wrist Straight"
            form_fault = True
        # 7. Arm Plane Check (curling out to the side)
        # During curl, wrist should be in front of shoulder (smaller z-value)
        elif stage == 'up' and wrist_lm.z > shoulder_lm.z:
            feedback = "Curl in Front of Body"
            form_fault = True

        # Log the first detected error in a rep attempt
        if form_fault and feedback and not self.last_error[1] == feedback:
            self.last_error = (arm_side, feedback)
            self.new_error_logged = True

        # --- State Machine for Rep Counting ---
        if elbow_angle > 160:  # Arm is in the "down" position
            if stage == 'up':  # rep completed
                # Check for partial contraction at the end of the 'up' phase
                if min_angle > 60:
                    feedback = "Curl Higher"
                    self.last_error = (arm_side, feedback)
                    self.new_error_logged = True
                # Reset for next rep
                min_angle = 180
            stage = "down"
        elif elbow_angle < 30 and stage == "down":  # Arm is in the "up" position
            # Check for partial extension before starting the 'up' phase
            if max_angle < 160:
                feedback = "Straighten Arm at Bottom"
                form_fault = True

            # Tempo Check
            elif rep_time > 0 and time.time() - rep_time < 0.8:
                feedback = "Too Fast!"
                form_fault = True

            if form_fault and feedback and not self.last_error[1] == feedback:
                self.last_error = (arm_side, feedback)
                self.new_error_logged = True

            stage = "up"
            rep_time = time.time()
            max_angle = 0  # Reset max angle for next extension check

            if not form_fault:
                counter += 1
                print(f"{arm_side.capitalize()} Reps: {counter}")

        # Track min/max angles during the rep
        if stage == 'up':
            min_angle = min(min_angle, elbow_angle)
        elif stage == 'down':
            max_angle = max(max_angle, elbow_angle)

        # If a fault was triggered, show the feedback. Otherwise, clear it.
        final_feedback = f"{arm_side[0].upper()}: {feedback}" if feedback else ""

        # --- Update attributes ---
        if arm_side == 'left':
            self.left_counter, self.left_stage, self.left_form_fault, self.left_feedback, self.left_angle, self.left_rep_time = counter, stage, form_fault, final_feedback, elbow_angle, rep_time
            self.left_min_angle, self.left_max_angle = min_angle, max_angle
        else:
            self.right_counter, self.right_stage, self.right_form_fault, self.right_feedback, self.right_angle, self.right_rep_time = counter, stage, form_fault, final_feedback, elbow_angle, rep_time
            self.right_min_angle, self.right_max_angle = min_angle, max_angle

    def process(self, image, landmarks):
        try:
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
            left_index = [lm[lm_pose.LEFT_INDEX.value].x,
                          lm[lm_pose.LEFT_INDEX.value].y]
            self._process_arm(left_shoulder, left_elbow,
                              left_wrist, left_hip, left_index, 'left', lm)

            # Right Arm
            right_shoulder = [lm[lm_pose.RIGHT_SHOULDER.value].x,
                              lm[lm_pose.RIGHT_SHOULDER.value].y]
            right_elbow = [lm[lm_pose.RIGHT_ELBOW.value].x,
                           lm[lm_pose.RIGHT_ELBOW.value].y]
            right_wrist = [lm[lm_pose.RIGHT_WRIST.value].x,
                           lm[lm_pose.RIGHT_WRIST.value].y]
            right_hip = [lm[lm_pose.RIGHT_HIP.value].x,
                         lm[lm_pose.RIGHT_HIP.value].y]
            right_index = [lm[lm_pose.RIGHT_INDEX.value].x,
                           lm[lm_pose.RIGHT_INDEX.value].y]
            self._process_arm(right_shoulder, right_elbow,
                              right_wrist, right_hip, right_index, 'right', lm)
        except (IndexError, KeyError):
            print("Warning: Not all required landmarks are visible.")
