import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
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

        # For summary: list of sets, one per rep, containing error tags
        self.left_rep_issues = []
        self.right_rep_issues = []

        # For persistent error tracking during a single rep
        self.left_current_rep_errors = set()
        self.right_current_rep_errors = set()

        # For smoothness/jerk detection
        self.left_angle_history = deque(maxlen=10)
        self.right_angle_history = deque(maxlen=10)

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
        # This is where completed rep issues are stored
        rep_issues_list = self.left_rep_issues if arm_side == 'left' else self.right_rep_issues

        current_rep_errors = set()

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
        error_tag = None
        if not (165 < torso_angle < 195):
            feedback = "Don’t Lean Back"
            form_fault = True
            error_tag = "leaned_back"
        elif shoulder_angle > 35:
            feedback = "Keep Elbow Still"
            form_fault = True
            error_tag = "elbow_moved"
        elif elbow[1] < shoulder[1]:
            feedback = "Keep Elbow Down"
            form_fault = True
            error_tag = "elbow_raised"
        elif elbow_angle > 190:
            feedback = "Don’t Overextend"
            form_fault = True
            error_tag = "overextended"
        elif abs(elbow[0] - shoulder[0]) > 0.1:
            feedback = "Keep Elbows Tucked In"
            form_fault = True
            error_tag = "elbows_out"
        elif elbow_angle < 120 and not (150 < wrist_angle < 210):
            feedback = "Keep Wrist Straight"
            form_fault = True
            error_tag = "bent_wrist"
        elif stage == 'up' and wrist_lm.z > shoulder_lm.z:
            feedback = "Curl in Front of Body"
            form_fault = True
            error_tag = "behind_body_curl"

        if form_fault and feedback and self.last_error[1] != feedback:
            self.last_error = (arm_side, feedback)
            self.new_error_logged = True

        if error_tag:
            # Add error to the persistent set for the current rep
            (self.left_current_rep_errors if arm_side ==
             'left' else self.right_current_rep_errors).add(error_tag)

        # --- Rep counting logic (relaxed for realistic motion) ---
        if elbow_angle > 160:  # down phase
            if stage == 'up':  # rep completed
                if min_angle > 70:  # was 60 — slightly relaxed
                    feedback = "Curl Higher"
                    # ✅ FIX: Add to the persistent error set for the correct arm
                    (self.left_current_rep_errors if arm_side == 'left'
                     else self.right_current_rep_errors).add("shallow_curl")
                    self.last_error = (arm_side, feedback)
                    self.new_error_logged = True
                min_angle = 180
                # At rep completion, store the errors for this rep
                # ✅ Store a COPY of the errors collected during this rep
                rep_issues_list.append(set(
                    self.left_current_rep_errors if arm_side == 'left' else self.right_current_rep_errors))
                # ✅ FIX: Clear the persistent error set for the correct arm after the rep is logged.
                if arm_side == 'left':
                    self.left_current_rep_errors.clear()
                else:
                    self.right_current_rep_errors.clear()
            stage = "down"

        elif elbow_angle < 45 and stage == "down":  # up phase (was <30)
            if max_angle < 150:
                feedback = "Straighten Arm at Bottom"
                form_fault = True
                current_rep_errors.add("incomplete_extension")
            elif rep_time > 0 and time.time() - rep_time < 0.7:  # faster check
                feedback = "Too Fast!"
                form_fault = True
                current_rep_errors.add("too_fast")

            if form_fault and feedback and self.last_error[1] != feedback:
                self.last_error = (arm_side, feedback)
                self.new_error_logged = True

            stage = "up"
            rep_time = time.time()
            max_angle = 0

            # ✅ Count rep even if minor faults occurred — user still moved fully
            counter += 1
            print(f"{arm_side.capitalize()} Reps: {counter}")

        # Track extremes for next rep
        if stage == 'up':
            min_angle = min(min_angle, elbow_angle)
        elif stage == 'down':
            max_angle = max(max_angle, elbow_angle)

        # Append angle to history for smoothness calculation
        if arm_side == 'left':
            self.left_angle_history.append(elbow_angle)
        else:
            self.right_angle_history.append(elbow_angle)

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
            'side': arm_side,
            'error_tag': error_tag
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
        """Generate realistic summary for bicep curls using rep-based accuracy."""
        total_time = time.time() - self.start_time if self.start_time else 0
        total_reps = min(self.left_counter, self.right_counter)

        # Handle missing reps
        if total_reps == 0 and len(self.recording) > 30:
            total_reps = 1  # motion fallback

        # --- Combine rep issues from both arms ---
        # Take the issues from the arm that completed the minimum number of reps
        num_left_reps = len(self.left_rep_issues)
        num_right_reps = len(self.right_rep_issues)
        rep_issues = self.left_rep_issues[:total_reps] if num_left_reps >= num_right_reps else self.right_rep_issues[:total_reps]

        # --- Compute accuracy ---
        faulty_reps = len([r for r in rep_issues if r])
        form_accuracy = 1.0 - (faulty_reps / max(1, total_reps))
        # mild penalty curve for realism
        form_accuracy = round((form_accuracy ** 1.25) * 100, 1)

        # --- Final feedback text ---
        avg_angle = np.mean([r["elbow_angle"]
                            for r in self.recording if r.get("elbow_angle")])
        if np.isnan(avg_angle):
            avg_angle = 0.0
        ideal_angle = 45.0
        diff_pct = ((avg_angle - ideal_angle) / ideal_angle) * 100
        feedback_text = (
            f"Try curling higher! Ideal bottom angle: {ideal_angle:.1f}°, "
            f"Your Avg: {avg_angle:.1f}° ({diff_pct:.1f}% above ideal)."
            if avg_angle > ideal_angle + 5
            else "Excellent range and control!"
        )

        # --- Smoothness metric (similar to squats) ---
        # Combine angle history from both arms for a total session score
        all_angles = list(self.left_angle_history) + \
            list(self.right_angle_history)
        smoothness_score = 1.0
        if len(all_angles) >= 10:
            vals = np.array(all_angles)
            # Calculate jerk (mean of the absolute second derivative)
            jerk = np.mean(np.abs(np.diff(np.diff(vals)))
                           ) if len(vals) > 3 else 0
            # Normalize jerk into a 0-1 score. Higher jerk = lower score.
            # ✅ FIX: Further increased divisor to make smoothness less sensitive for curl motions.
            smoothness_score = round(
                max(0.0, 1.0 - min(1.0, jerk / 150.0)), 3)

        # --- Issue count summary ---
        issue_counts = {}
        for s in rep_issues:
            for tag in s:
                issue_counts[tag] = issue_counts.get(tag, 0) + 1

        # --- Build summary dict ---
        summary = {
            "exercise": self.exercise_name,
            "total_reps": int(total_reps),
            "session_time_sec": int(total_time),
            "frames_recorded": len(self.recording),
            "form_accuracy": form_accuracy,
            "rep_issues": rep_issues,
            "feedback": feedback_text,
            "issue_counts": issue_counts,
            "smoothness_score": smoothness_score,
            "avg_angle": round(avg_angle, 1),
        }

        return summary

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
