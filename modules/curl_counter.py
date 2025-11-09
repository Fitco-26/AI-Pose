import cv2
import numpy as np
import mediapipe as mp
import time
from .posture.utils.angle_utils import calculate_angle, map_angle_to_progress

mp_pose = mp.solutions.pose


class CurlCounter:
    def __init__(self):
        # --- Left arm state ---
        self.left_counter = 0
        self.left_stage = 'down'
        self.left_feedback = ""
        self.left_form_fault = False
        self.left_angle = 160.0
        self.left_rep_time = 0
        self.left_min_angle = 180
        self.left_max_angle = 0

        # --- Right arm state ---
        self.right_counter = 0
        self.right_stage = 'down'
        self.right_feedback = ""
        self.right_form_fault = False
        self.right_angle = 160.0
        self.right_rep_time = 0
        self.right_min_angle = 180
        self.right_max_angle = 0

        # --- Error logging ---
        self.new_error_logged = False
        self.last_error = (None, None)

        # --- Recording for DB ---
        self.recording = []
        self.start_time = None
        self.exercise_name = "bicep_curls"

    # --------------------------------------------------------------------
    def _process_arm(self, shoulder, elbow, wrist, hip, index_finger, arm_side: str, landmarks):
        """Processes rep counting and form detection for one arm."""
        if any(p is None for p in [shoulder, elbow, wrist, hip, index_finger]):
            return  # skip if any point missing

        # Select side-specific variables
        counter = self.left_counter if arm_side == 'left' else self.right_counter
        stage = self.left_stage if arm_side == 'left' else self.right_stage
        form_fault = self.left_form_fault if arm_side == 'left' else self.right_form_fault
        rep_time = self.left_rep_time if arm_side == 'left' else self.right_rep_time
        min_angle = self.left_min_angle if arm_side == 'left' else self.right_min_angle
        max_angle = self.left_max_angle if arm_side == 'left' else self.right_max_angle

        feedback = ""

        # Calculate key angles
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        shoulder_angle = calculate_angle(hip, shoulder, elbow)
        wrist_angle = calculate_angle(elbow, wrist, index_finger)

        # Extract 3D positions for movement plane check
        lm_pose = mp_pose.PoseLandmark
        shoulder_lm = landmarks[lm_pose[f'{arm_side.upper()}_SHOULDER'].value]
        wrist_lm = landmarks[lm_pose[f'{arm_side.upper()}_WRIST'].value]

        # Torso verticality check
        torso_angle = calculate_angle([hip[0], hip[1] + 0.5], hip, shoulder)

        # --- Form fault checks ---
        form_fault = False
        if not (165 < torso_angle < 195):
            feedback = "Don’t Lean Back"
            form_fault = True
        elif shoulder_angle > 35:
            feedback = "Keep Elbow Still"
            form_fault = True
        elif elbow[1] < shoulder[1]:
            feedback = "Keep Elbow Down"
            form_fault = True
        elif elbow_angle > 190:
            feedback = "Don’t Overextend"
            form_fault = True
        elif abs(elbow[0] - shoulder[0]) > 0.1:
            feedback = "Keep Elbows Tucked In"
            form_fault = True
        elif elbow_angle < 120 and not (150 < wrist_angle < 210):
            feedback = "Keep Wrist Straight"
            form_fault = True
        elif stage == 'up' and wrist_lm.z > shoulder_lm.z:
            feedback = "Curl in Front of Body"
            form_fault = True

        if form_fault and feedback and self.last_error[1] != feedback:
            self.last_error = (arm_side, feedback)
            self.new_error_logged = True

        # --- Rep counting logic ---
        if elbow_angle > 160:  # down phase
            if stage == 'up':  # rep completed
                if min_angle > 60:
                    feedback = "Curl Higher"
                    self.last_error = (arm_side, feedback)
                    self.new_error_logged = True
                min_angle = 180
            stage = "down"
        elif elbow_angle < 30 and stage == "down":  # up phase
            if max_angle < 160:
                feedback = "Straighten Arm at Bottom"
                form_fault = True
            elif rep_time > 0 and time.time() - rep_time < 0.8:
                feedback = "Too Fast!"
                form_fault = True

            if form_fault and feedback and self.last_error[1] != feedback:
                self.last_error = (arm_side, feedback)
                self.new_error_logged = True

            stage = "up"
            rep_time = time.time()
            max_angle = 0

            if not form_fault:
                counter += 1
                print(f"{arm_side.capitalize()} Reps: {counter}")

        # Track extremes for next rep
        if stage == 'up':
            min_angle = min(min_angle, elbow_angle)
        elif stage == 'down':
            max_angle = max(max_angle, elbow_angle)

        # Prepare formatted feedback
        final_feedback = f"{arm_side[0].upper()}: {feedback}" if feedback else ""

        # Store updates
        if arm_side == 'left':
            self.left_counter, self.left_stage, self.left_form_fault, self.left_feedback = counter, stage, form_fault, final_feedback
            self.left_angle, self.left_rep_time = elbow_angle, rep_time
            self.left_min_angle, self.left_max_angle = min_angle, max_angle
        else:
            self.right_counter, self.right_stage, self.right_form_fault, self.right_feedback = counter, stage, form_fault, final_feedback
            self.right_angle, self.right_rep_time = elbow_angle, rep_time
            self.right_min_angle, self.right_max_angle = min_angle, max_angle

        # --- Record for analysis ---
        t = time.time() - self.start_time if self.start_time else 0
        rec = {
            't': t,
            'stage': stage,
            'form_fault': form_fault,
            'elbow_angle': elbow_angle,
            'side': arm_side
        }
        self.recording.append(rec)

    # --------------------------------------------------------------------
    def process(self, image, landmarks):
        """Process one Mediapipe pose frame for both arms (handles mirrored video)."""
        if landmarks is None:
            return

        try:
            lm = landmarks
            lm_pose = mp_pose.PoseLandmark

            def pt(idx):
                try:
                    p = lm[idx]
                    return [p.x, p.y]
                except Exception:
                    return None

        # ✅ Swap logic because of mirrored camera feed
        # What appears on screen as "right" is actually Mediapipe's LEFT_* landmarks
        # and vice versa.
        # So we intentionally cross-map the sides.

        # Screen-right (your physical right hand)
            right_shoulder = pt(lm_pose.LEFT_SHOULDER.value)
            right_elbow = pt(lm_pose.LEFT_ELBOW.value)
            right_wrist = pt(lm_pose.LEFT_WRIST.value)
            right_hip = pt(lm_pose.LEFT_HIP.value)
            right_index = pt(lm_pose.LEFT_INDEX.value)
            self._process_arm(right_shoulder, right_elbow, right_wrist,
                              right_hip, right_index, 'right', lm)

        # Screen-left (your physical left hand)
            left_shoulder = pt(lm_pose.RIGHT_SHOULDER.value)
            left_elbow = pt(lm_pose.RIGHT_ELBOW.value)
            left_wrist = pt(lm_pose.RIGHT_WRIST.value)
            left_hip = pt(lm_pose.RIGHT_HIP.value)
            left_index = pt(lm_pose.RIGHT_INDEX.value)
            self._process_arm(left_shoulder, left_elbow,
                              left_wrist, left_hip, left_index, 'left', lm)

            if self.start_time is None:
                self.start_time = time.time()

        except Exception as e:
            print("⚠️ Error in CurlCounter.process:", e)

    # --------------------------------------------------------------------
    def get_session_summary(self):
        """Returns session summary for DB storage."""
        total_time = time.time() - self.start_time if self.start_time else 0
        total_reps = min(self.left_counter, self.right_counter)
        feedbacks = [f for f in [self.left_feedback, self.right_feedback] if f]
        final_feedback = " | ".join(
            sorted(feedbacks)) if feedbacks else "Good form!"

        return {
            'exercise': self.exercise_name,
            'total_reps': total_reps,
            'session_time_sec': int(total_time),
            'feedback': final_feedback,
            'frames_recorded': len(self.recording)
        }

    # --------------------------------------------------------------------
    def get_stats(self):
        """Live stats for real-time updates."""
        progress = 0
        if self.left_stage == 'up':
            progress = map_angle_to_progress(self.left_angle)
        elif self.right_stage == 'up':
            progress = map_angle_to_progress(self.right_angle)

        return {
            "left": self.left_counter,
            "right": self.right_counter,
            "total": min(self.left_counter, self.right_counter),
            "stage": f"L-{self.left_stage} | R-{self.right_stage}",
            "warning": self.left_feedback or self.right_feedback or "",
            "progress": progress,
            "error_log": []
        }

    # --------------------------------------------------------------------
    def release(self):
        """Clean up resources."""
        pass
