import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import mysql.connector
import threading
import time
import csv
import os
from collections import deque
from modules.posture.utils.angle_utils import calculate_angle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class FrameGrabber:
    """
    Threaded camera reader to reduce camera open latency and provide the latest frame quickly.
    Usage:
        grabber = FrameGrabber(src=0)
        grabber.start()
        frame = grabber.read()  # latest frame (or None)
        grabber.stop()
    """

    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(
            src, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # optional: lower buffer latency
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._stopped = True
        self.frame = None
        self.lock = threading.Lock()
        self.thread = None

    def _grab_loop(self):
        # Warm-up time to let camera auto-exposure settle
        warmup_start = time.time()
        while not self._stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
            # after a short warmup, keep reading as fast as possible
            if time.time() - warmup_start < 0.5:
                time.sleep(0.01)

    def start(self):
        if self.thread is None:
            self._stopped = False
            self.thread = threading.Thread(target=self._grab_loop, daemon=True)
            self.thread.start()
            # small wait for first frames to arrive
            t0 = time.time()
            while self.read() is None and time.time() - t0 < 1.0:
                time.sleep(0.01)
        return self

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self._stopped = True
        if self.thread is not None:
            self.thread.join(timeout=0.5)
        try:
            self.cap.release()
        except Exception:
            pass
        self.thread = None


class SquatCorrector:
    """
    Optimized SquatCorrector with:
      - session recording
      - richer error detection
      - get_session_summary() for separate states screen
      - analyze_session() for post-hoc analysis
    """

    def __init__(self, speak_feedback=True, exercise_name="squat"):
        # MediaPipe Pose
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.6, min_tracking_confidence=0.6)

        # Session & state
        self.exercise_name = exercise_name
        self.rep_counter = 0
        self.squat_stage = 'standing'
        self.form_feedback = ""
        self.form_fault = False
        self.depth_achieved = False

        # Recording buffer for session (list of dict per frame)
        # appended dicts: {t, knee_angle, back_angle, stage, errors}
        self.recording = []
        self.start_time = None

        # For smoothness/jerk detection: store recent knee angles
        self.knee_history = deque(maxlen=8)
        self.last_knee_angle = None

        # Feedback timing and TTS
        self.speak_feedback = speak_feedback
        self.last_feedback_time = 0
        self.feedback_cooldown = 2.5  # seconds
        if self.speak_feedback:
            self.tts_engine = pyttsx3.init()

        # Thread-limiting for TTS
        self._tts_lock = threading.Lock()

        # Visibility threshold for deciding if side view is reliable
        self.visibility_threshold = 0.4

        # Errors aggregated per rep (for summary)
        self.rep_issues = []  # list of sets, one per counted rep

        # For live error logging, similar to CurlCounter
        self.new_error_logged = False
        self.last_error_message = None
        self.last_error = (None, None)

    # ---------------- Internal helpers ----------------

    def _speak_async(self, text):
        if not self.speak_feedback or not text:
            return

        def run():
            try:
                with self._tts_lock:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
            except Exception as e:
                print("TTS Error:", e)

        # start daemon thread (limit concurrency implicitly via lock)
        threading.Thread(target=run, daemon=True).start()

    def _can_give_feedback(self):
        return time.time() - self.last_feedback_time > self.feedback_cooldown

    # Extract landmarks for side with fallback
    def _extract_landmarks(self, landmarks):
        left_vis = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
        right_vis = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
        use_left = left_vis >= right_vis

        # if both low visibility, still choose left but mark visibility low
        side = "LEFT" if use_left else "RIGHT"

        def lm(name):
            return landmarks[getattr(mp_pose.PoseLandmark, name).value]

        shoulder = [lm(f"{side}_SHOULDER").x, lm(
            f"{side}_SHOULDER").y, lm(f"{side}_SHOULDER").visibility]
        hip = [lm(f"{side}_HIP").x, lm(f"{side}_HIP").y,
               lm(f"{side}_HIP").visibility]
        knee = [lm(f"{side}_KNEE").x, lm(f"{side}_KNEE").y,
                lm(f"{side}_KNEE").visibility]
        ankle = [lm(f"{side}_ANKLE").x, lm(f"{side}_ANKLE").y,
                 lm(f"{side}_ANKLE").visibility]
        foot_index = [lm(f"{side}_FOOT_INDEX").x, lm(
            f"{side}_FOOT_INDEX").y, lm(f"{side}_FOOT_INDEX").visibility]

        # also try to get the opposite knee for asymmetry checks if visible
        other = "RIGHT" if side == "LEFT" else "LEFT"
        try:
            other_knee = [lm(f"{other}_KNEE").x, lm(
                f"{other}_KNEE").y, lm(f"{other}_KNEE").visibility]
        except Exception:
            other_knee = [None, None, 0.0]

        visibility_score = np.mean(
            [shoulder[2] or 0, hip[2] or 0, knee[2] or 0, ankle[2] or 0, foot_index[2] or 0])
        return shoulder, hip, knee, ankle, foot_index, other_knee, visibility_score

    # ---------------- Error detectors (squat-specific) ----------------

    def _detect_errors(self, back_angle, knee_coord, foot_coord, knee_angle, visibility_score):
        """
        Returns a set of error tags and textual messages (a short primary message).
        Tags: 'back_bent', 'knee_over_toe', 'shallow', 'incomplete_extension', 'jerky', 'asymmetry', 'poor_visibility'
        """
        errors = set()
        primary_msg = None

        # visibility
        if visibility_score < self.visibility_threshold:
            errors.add('poor_visibility')
            primary_msg = "Move back so your full body is visible."

        # back posture
        if back_angle < 145:  # Relaxed from 155
            errors.add('back_bent')
            if not primary_msg:
                primary_msg = "Keep your back straight!"

        # knee over toe (x-coords): small tolerance
        if knee_coord[0] > foot_coord[0] + 0.04:  # Relaxed from 0.02
            errors.add('knee_over_toe')
            if not primary_msg:
                primary_msg = "Keep knees behind your toes!"

        # depth / shallow: depth flag checked elsewhere; here check thresholds
        if not self.depth_achieved and knee_angle > 100 and knee_angle < 150:
            # user is partially bent but not reaching depth yet; flag shallow late in rep
            errors.add('shallow')
            if not primary_msg:
                primary_msg = "Try lowering until hips are below knees."

        # incomplete extension at top (knee never > 160 during rep)
        # we'll handle this at rep-complete time by checking rep_issues

        # jerkiness: compute variance of recent knee changes (higher => jerky)
        if len(self.knee_history) >= 4:
            vals = np.array(self.knee_history)
            # approximate jerk = mean absolute second derivative
            d1 = np.diff(vals)
            d2 = np.diff(d1)
            jerk = np.mean(np.abs(d2)) if len(d2) > 0 else 0
            if jerk > 25:  # heuristic threshold
                errors.add('jerky')
                if not primary_msg:
                    primary_msg = "Control your speed — try a smoother motion."

        # asymmetry: needs other knee visible; possible only if both visible (rare in side view)
        # We check difference in knee x positions or y positions to detect lateral shift
        # (This is a weak test from one camera, but better than nothing.)
        # Asymmetry detection will be computed in analyze_session if we have per-frame other knee coords.

        # fallback primary message
        if not primary_msg and errors:
            primary_msg = "Adjust your form."

        return errors, primary_msg

    # ---------------- State & rep update logic ----------------

    def _update_rep_state_and_record(self, t, knee_angle, back_angle, shoulder, hip, knee, ankle, foot_index, visibility_score, other_knee):
        """
        State machine refined. Handles rep counting and per-rep issue aggregation.
        Also records per-frame metrics.
        """
        # keep knee history for jerk detection
        if knee_angle is not None:
            self.knee_history.append(knee_angle)

        # detect errors at this frame
        errors, primary_msg = self._detect_errors(
            back_angle, knee, foot_index, knee_angle, visibility_score)
        if primary_msg:
            # only speak if cooldown passed
            if self._can_give_feedback():
                self._speak_async(primary_msg)
                self.last_feedback_time = time.time()

        # Save frame-level record
        rec = {
            't': t,
            'knee_angle': float(knee_angle) if knee_angle is not None else None,
            'back_angle': float(back_angle) if back_angle is not None else None,
            'stage': self.squat_stage,
            'errors': list(errors),
            'visibility': float(visibility_score)
        }
        # include other knee if available
        if other_knee and other_knee[2] and other_knee[2] > 0.2:
            rec['other_knee'] = other_knee[:2]

        self.recording.append(rec)

        # state transitions with clearer hysteresis
        if self.squat_stage == 'standing':
            if knee_angle < 150:  # Relaxed from 155
                self.squat_stage = 'descending'

        elif self.squat_stage == 'descending':
            if knee_angle <= 110:  # Relaxed from 105
                self.squat_stage = 'bottom'
            elif knee_angle > 165:
                self.squat_stage = 'standing'  # aborted

        elif self.squat_stage == 'bottom':
            # user holds bottom or starts ascending
            if knee_angle >= 115:  # Relaxed from 110
                self.squat_stage = 'ascending'

        elif self.squat_stage == 'ascending':
            if knee_angle >= 160:  # Relaxed from 165
                # rep complete -> evaluate issues collected during this rep
                # For now, check if depth_achieved and recorded errors during this rep
                # find frames belonging to last rep window: simple heuristic
                rep_frames = [r for r in self.recording[-40:] if r['stage']
                              in ('descending', 'bottom', 'ascending', 'standing')]
                rep_error_tags = set()
                for r in rep_frames:
                    rep_error_tags.update(r.get('errors', []))
                # incomplete extension: if no frame in rep had knee_angle > 160
                max_knee = max((r['knee_angle'] or 0) for r in rep_frames)
                if max_knee < 160:
                    rep_error_tags.add('incomplete_extension')

                # incomplete depth: if no frame with hip below knee
                depth_flag = any(
                    (r.get('knee_angle') is not None and r['knee_angle'] <= 110) for r in rep_frames)
                if not depth_flag:
                    rep_error_tags.add('shallow')

                # record rep issues
                self.rep_issues.append(rep_error_tags)

                # --- Live Error Logging ---
                # If there were errors in this rep, log them for the UI
                if rep_error_tags:
                    error_messages = []
                    if 'back_bent' in rep_error_tags:
                        error_messages.append("Bent Back")
                    if 'knee_over_toe' in rep_error_tags:
                        error_messages.append("Knees Over Toes")
                    if 'shallow' in rep_error_tags:
                        error_messages.append("Shallow Depth")
                    if 'jerky' in rep_error_tags:
                        error_messages.append("Jerky Motion")
                    if 'incomplete_extension' in rep_error_tags:
                        error_messages.append("Incomplete Extension")

                    if error_messages:
                        # This will be picked up by app.py
                        self.last_error_message = ", ".join(error_messages)
                        self.new_error_logged = True
                        # Also expose a tuple for app.py compatibility (same as CurlCounter)
                        self.last_error = ('squat', self.last_error_message)

                # Count rep only if no critical safety issues and depth achieved
                # Rep is counted if depth is achieved, feedback is given separately.
                if depth_flag:
                    self.rep_counter += 1
                    # speak rep count
                    if self.speak_feedback:
                        self._speak_async(str(self.rep_counter))
                else:
                    # speak encouraging correction
                    if 'knee_over_toe' in rep_error_tags and self._can_give_feedback():
                        self._speak_async(
                            "Avoid pushing knees forward beyond toes.")
                        self.last_feedback_time = time.time()
                    elif 'back_bent' in rep_error_tags and self._can_give_feedback():
                        self._speak_async(
                            "Keep your chest up and back straight.")
                        self.last_feedback_time = time.time()
                    else:
                        # general feedback
                        if self._can_give_feedback():
                            self._speak_async(
                                "Try to go deeper and keep control.")
                            self.last_feedback_time = time.time()

                # reset flags
                self.depth_achieved = False
                self.knee_history.clear()
                self.squat_stage = 'standing'

        # depth check (hip below knee in screen y)
        if self.squat_stage in ['descending', 'bottom']:
            if hip[1] >= knee[1]:
                self.depth_achieved = True

    # ---------------- Main public API ----------------

    def process_frame(self, frame, draw_landmarks=True):
        """
        Processes a single frame; returns (image, rep_count, stage, form_feedback).
        Also records per-frame metrics into self.recording.
        """
        if self.start_time is None:
            self.start_time = time.time()
        t = time.time() - self.start_time

        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # default feedback (persisted)
        # do not clear self.form_feedback immediately so UI can show it
        visibility_score = 0.0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder, hip, knee, ankle, foot_index, other_knee, visibility_score = self._extract_landmarks(
                landmarks)

            # compute angles (guard if any coordinate is None)
            try:
                knee_angle = calculate_angle(hip[:2], knee[:2], ankle[:2])
                back_angle = calculate_angle(shoulder[:2], hip[:2], knee[:2])
            except Exception:
                knee_angle = None
                back_angle = None

            # compute errors, record, and update state machine
            self._update_rep_state_and_record(
                t, knee_angle, back_angle, shoulder, hip, knee, ankle, foot_index, visibility_score, other_knee)

            # Visual aids: draw pose
            if draw_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(
                        color=(255, 0, 255), thickness=2, circle_radius=2)
                )

            # Generate a friendly top-level feedback for UI (aggregate recent error)
            recent_errors = set()
            if len(self.recording) > 0:
                recent_errors.update(*(r.get('errors', [])
                                     for r in self.recording[-10:]))
            # choose priority message
            priority_msgs = {
                'poor_visibility': "Move back so your whole body is visible.",
                'back_bent': "Keep your back straight.",
                'knee_over_toe': "Don't let your knee go past toes.",
                'jerky': "Control your speed — smoother motion.",
                'shallow': "Try lowering until hips go below knees.",
                'incomplete_extension': "Fully extend at top for full reps."
            }
            # pick highest priority error if any
            ui_msg = ""
            for tag in ['poor_visibility', 'back_bent', 'knee_over_toe', 'jerky', 'shallow', 'incomplete_extension']:
                if tag in recent_errors:
                    ui_msg = priority_msgs[tag]
                    break
            self.form_feedback = ui_msg

        else:
            # No pose detected
            self.form_feedback = "No pose detected. Please ensure full body is in frame."
        return image, self.rep_counter, self.squat_stage, self.form_feedback

    # ---------------- Session summary and analysis ----------------

    def get_session_summary(self):
        """Generate realistic summary for squat sessions using rep-based accuracy."""
        total_time = time.time() - self.start_time if self.start_time else 0
        total_reps = self.rep_counter

        # Handle fallback for motion without full rep
        if total_reps == 0 and len(self.recording) > 30:
            total_reps = 1

        # --- Compute form accuracy ---
        # ✅ FIX: Use the reliable self.rep_issues list directly.
        # Calculate accuracy based on total errors vs. total reps, which is more granular.
        # A rep can have multiple errors. We penalize based on the total error count.
        total_errors = sum(len(s) for s in self.rep_issues)
        # Create a score where each error slightly reduces the total accuracy.
        # The penalty per error is scaled by the number of reps.
        # This prevents a single bad rep from destroying the score in a long set.
        # 5.0 is a tunable factor
        accuracy_score = 1.0 - (total_errors / (max(1, total_reps) * 5.0))
        # Clamp score between 0 and 1 and convert to percentage
        form_accuracy = round(max(0.0, min(1.0, accuracy_score)) * 100, 1)

        # --- Smoothness metric ---
        all_knees = [r["knee_angle"]
                     for r in self.recording if r.get("knee_angle") is not None]
        smoothness_score = 1.0
        if len(all_knees) >= 6:
            vals = np.array(all_knees)
            jerk = np.mean(np.abs(np.diff(np.diff(vals)))
                           ) if len(vals) > 3 else 0
            smoothness_score = round(
                max(0.0, 1.0 - min(1.0, jerk / 50.0)), 3)

        # --- Compute depth feedback ---
        avg_angle = np.mean(all_knees) if len(all_knees) else 0
        if np.isnan(avg_angle):
            avg_angle = 0.0
        ideal_angle = 100.0
        depth_pct = ((avg_angle - ideal_angle) / ideal_angle) * 100
        feedback_text = (
            f"Your average squat depth is {depth_pct:.1f}% shallower than ideal. "
            f"Aim for around {ideal_angle:.1f}° at the bottom."
            if avg_angle > ideal_angle + 5
            else "Excellent depth and form!"
        )

        # --- Issue count summary ---
        issue_counts = {}
        for s in self.rep_issues:
            for tag in s:
                issue_counts[tag] = issue_counts.get(tag, 0) + 1

        # --- Final summary dict ---
        summary = {
            "exercise": self.exercise_name,
            "total_reps": int(total_reps),
            "session_time_sec": int(total_time),
            "form_accuracy": form_accuracy,
            "smoothness_score": smoothness_score,
            "frames_recorded": len(self.recording),
            "rep_issues": self.rep_issues,  # Use the correct list
            "issue_counts": issue_counts,
            "feedback": feedback_text,
            "avg_angle": round(avg_angle, 1),
        }

        return summary

    def analyze_session(self, out_csv_path=None):
        """
        Run a deterministic post-session analysis (fast). Exports CSV if path provided.
        Later you can replace or supplement this with the LSTM model.
        Returns summary dict (same as get_session_summary).
        """
        summary = self.get_session_summary()
        if out_csv_path:
            try:
                keys = ['t', 'knee_angle', 'back_angle', 'stage',
                        'errors', 'visibility', 'other_knee']
                with open(out_csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    for r in self.recording:
                        row = {k: r.get(k, '') for k in keys}
                        # convert lists to strings for CSV
                        row['errors'] = ";".join(row['errors']) if isinstance(
                            row.get('errors'), list) else row.get('errors', '')
                        writer.writerow(row)
            except Exception as e:
                print("CSV export error:", e)
        return summary

    # ---------------- Cleanup ----------------

    def release(self):
        try:
            self.pose.close()
        except Exception:
            pass
        if self.speak_feedback:
            try:
                self.tts_engine.stop()
            except Exception:
                pass
