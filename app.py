import torch
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
from ultralytics import YOLO
from modules.pose_detector import PoseDetector
from modules.curl_counter import CurlCounter
from modules.squat_corrector import SquatCorrector
from modules.posture.utils.angle_utils import map_angle_to_progress
import mediapipe as mp
import pyttsx3
import threading
import time
from datetime import datetime, timedelta
import mysql.connector
import atexit
import json
import csv
import os
import queue

app = Flask(__name__)
# Necessary for session management
app.secret_key = 'your_very_secret_key_for_sessions'

# --- MySQL Database Connection ---
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Mokshith15@",
    database="fitness_tracker"
)
cursor = db.cursor(dictionary=True)


# --- Config Management (with Profiles) ---
CONFIG_FILE = 'config.json'
DEFAULT_CONFIG = {
    "speak_feedback": True,
    "default_target_reps": 15,
    "feedback_cooldown_sec": 5,
    "camera_index": 0,
    "user_name": "User",
    "ask_camera_permission": False  # Added from previous change
}


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_FILE, 'r') as f:
            return {**DEFAULT_CONFIG, **json.load(f)}  # Merge with defaults
    except (json.JSONDecodeError, IOError):
        return DEFAULT_CONFIG


def save_config(config_data):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, indent=4)


app.config['APP_CONFIG'] = load_config()

# --- TTS Engine Setup ---
tts_engine = pyttsx3.init()
speech_queue = queue.Queue()
tts_lock = threading.Lock()


def tts_worker():
    """Runs continuously in a background thread to handle queued speech."""
    while True:
        text = speech_queue.get()  # Wait for text
        if text is None:  # Exit signal
            break
        with tts_lock:
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
        speech_queue.task_done()


# Start the single background TTS thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()


def speak_async(text):
    """Adds speech text to the queue instead of creating new threads."""
    if not app.config['APP_CONFIG'].get('speak_feedback', True):
        return
    if text and isinstance(text, str):
        speech_queue.put(text)


def calculate_bounded_improvement(current, previous):
    """
    Computes a smooth, bounded improvement percentage between sessions,
    clamped between -20% and +20% for a more user-friendly display.
    This avoids extreme percentages when the previous score was very low.
    """
    if previous is None or previous <= 0:
        return 0.0

    # The change is the difference between the two accuracy scores (0-100).
    delta = current - previous
    # Normalize the change to a -20% to +20% range for display.
    improvement = max(-20.0, min(20.0, (delta / 100.0) * 20.0))
    return round(improvement, 2)


# --- Camera Management ---
_camera_instance = None
_camera_lock = threading.Lock()
_active_clients = 0
_camera_active = False


def get_camera():
    """Initializes or returns the global camera instance."""
    global _camera_instance, _active_clients, _camera_active

    with _camera_lock:
        # If camera is None or not opened, reinitialize
        if _camera_instance is None or not hasattr(_camera_instance, "isOpened") or not _camera_instance.isOpened():
            print("♻️ Reinitializing camera...")
            camera_index = app.config['APP_CONFIG'].get('camera_index', 0)
            _camera_instance = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            time.sleep(0.5)  # give hardware a moment
            if not _camera_instance.isOpened():
                print("❌ Error: Could not open video stream.")
                _camera_instance = None
                _camera_active = False
                return None
            else:
                _camera_active = True
                print("🎥 Camera opened successfully (reinitialized).")

        _active_clients += 1
        return _camera_instance


def release_camera():
    """Releases the global camera instance."""
    global _camera_instance, _active_clients, _camera_active
    with _camera_lock:
        _camera_active = False  # ✅ Signal frame loop to stop
        _active_clients = max(0, _active_clients - 1)
        if _active_clients == 0 and _camera_instance is not None:
            _camera_instance.release()
            _camera_instance = None
            print("✅ Camera fully released at hardware level.")

            # ✅ Additional hardware cleanup
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                print("🧹 OpenCV windows and buffers destroyed.")
            except Exception as e:
                print("⚠️ OpenCV cleanup failed:", e)

            # ✅ Force backend reset (final hardware handle release)
            try:
                tmp_cap = cv2.VideoCapture(
                    app.config['APP_CONFIG'].get('camera_index', 0))
                tmp_cap.release()
                print("🔁 Verified hardware handle released.")
            except Exception as e:
                print("⚠️ Verification camera release failed:", e)


# Register cleanup to ensure camera is released on app exit
atexit.register(release_camera)


# --- Initialize state ---
detector = PoseDetector()
counter = CurlCounter()
# Explicitly set the device for YOLO. It will use GPU if available, otherwise CPU.
# This ensures you're leveraging the GPU when possible.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO("models/best.pt").to(device)
is_workout_active = False
current_exercise = None      # To store the selected exercise ID
exercise_handler = None      # To hold the instance of CurlCounter or SquatCorrector

# Store stats for AJAX polling
stats = {
    "left": 0,
    "right": 0,
    "total": 0,
    "stage": "L- | R-",
    "warning": "",
    "progress": 0,
    "error_log": []
}

# State for voice feedback
last_spoken_feedback = ""
last_spoken_time = 0
last_spoken_rep = 0
workout_start_time = None
show_landmarks = True  # Global state for landmark visibility
ENABLE_CSV_BACKUP = False  # Optional CSV backup (disabled by default)
# --- History File ---
HISTORY_FILE = 'data/workout_history.csv'

# --- Landmark Visibility Check ---


def are_landmarks_visible(landmarks, visibility_threshold=0.5):
    """
    Checks if essential landmarks for bicep curls are visible.
    Lowered threshold to 0.5 to make detection more robust,
    especially for partial right-arm occlusions.
    """
    required_landmarks = [
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
        mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
        mp.solutions.pose.PoseLandmark.LEFT_WRIST,
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
        mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    ]
    for landmark_index in required_landmarks:
        if landmarks.landmark[landmark_index.value].visibility < visibility_threshold:
            return False
    return True

# --- Frame Generator ---


def generate_frames():
    global stats, is_workout_active, last_spoken_feedback, last_spoken_time, last_spoken_rep, exercise_handler, current_exercise, _camera_active

    # ensure form_warning exists each loop iteration
    form_warning = ""

    while True:
        if not _camera_active:
            print("🛑 Stopping frame generation — camera released.")
            try:
                if _camera_instance:
                    _camera_instance.release()
                    print("🔒 Camera capture forcibly closed inside generator.")
            except Exception as e:
                print("⚠️ Camera cleanup inside generator failed:", e)
            break

        try:
            cap = get_camera()
            if not _camera_active or cap is None or not cap.isOpened():
                print("⚠️ Camera not ready, retrying initialization...")
                time.sleep(0.2)
                # Send a placeholder frame
                cap = get_camera()
                if cap is None or not cap.isOpened():
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, "Camera not available", (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                _, buffer = cv2.imencode('.jpg', blank_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame. Retrying...")
                time.sleep(0.1)
                continue

            # 🔄 Flip the frame *before* detection for correct handedness
            frame = cv2.flip(frame, 1)

            processed_image = frame.copy()

            # --- Always-on Pose Detection and Landmark Drawing ---
            processed_image, results = detector.detect(processed_image)
            landmarks_visible = results.pose_landmarks and are_landmarks_visible(
                results.pose_landmarks)

            # Draw landmarks if the toggle is on, regardless of workout state.
            if show_landmarks:
                detector.draw_landmarks(processed_image, results)

            # ✅ Optional: draw debug elbow angles on screen
            if is_workout_active and isinstance(exercise_handler, CurlCounter) and results.pose_landmarks:
                cv2.putText(processed_image, f"L:{int(exercise_handler.left_angle)}°",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(processed_image, f"R:{int(exercise_handler.right_angle)}°",
                            (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if is_workout_active:
                # --- Bicep Curls Logic ---
                if current_exercise == 'bicep_curls' and isinstance(exercise_handler, CurlCounter):
                    yolo_results = yolo_model(frame, verbose=False)

                    if landmarks_visible:
                        exercise_handler.process(
                            frame, results.pose_landmarks.landmark)  # This method doesn't modify the image
                        left, right = exercise_handler.left_counter, exercise_handler.right_counter
                        total = min(left, right)
                        stats["left"], stats["right"], stats["total"] = left, right, total
                        stats["stage"] = f"L-{exercise_handler.left_stage or '-'} | R-{exercise_handler.right_stage or '-'}"

                        feedbacks = [f for f in [
                            exercise_handler.left_feedback, exercise_handler.right_feedback] if f]
                        if feedbacks:
                            form_warning = " | ".join(sorted(feedbacks))

                        if exercise_handler.new_error_logged:
                            # Unified error handling for both Curl and Squat
                            if hasattr(exercise_handler, 'last_error'):
                                error_side, error_msg = exercise_handler.last_error
                                rep_at_error = stats.get(
                                    error_side, stats.get('total', 0)) + 1
                                side_text = f" ({error_side.capitalize()})" if error_side != 'squat' else ""
                                stats["error_log"].append(
                                    f"At Rep {rep_at_error}{side_text}: {error_msg}")
                                exercise_handler.new_error_logged = False

                        current_rep_count = max(left, right)
                        if current_rep_count > last_spoken_rep:
                            speak_async(str(current_rep_count))
                            last_spoken_rep = current_rep_count

                            if current_rep_count == round(TARGET_REPS / 2):
                                speak_async("Halfway there, keep it up!")
                            elif current_rep_count == TARGET_REPS:
                                speak_async("Great set! Take a rest!")

                        # 🧮 Balanced progress logic: handles both arms fairly
                        if exercise_handler.left_stage == 'up' and exercise_handler.right_stage == 'up':
                            stats["progress"] = (
                                map_angle_to_progress(exercise_handler.left_angle) +
                                map_angle_to_progress(
                                    exercise_handler.right_angle)
                            ) / 2
                        elif exercise_handler.left_stage == 'up':
                            stats["progress"] = map_angle_to_progress(
                                exercise_handler.left_angle)
                        elif exercise_handler.right_stage == 'up':
                            stats["progress"] = map_angle_to_progress(
                                exercise_handler.right_angle)
                        else:
                            stats["progress"] = 0

                    elif results.pose_landmarks:
                        form_warning = "Please make sure your full upper body is visible."

                    dumbbell_detected = any(
                        len(r.boxes) > 0 for r in yolo_results)

                    if not dumbbell_detected:
                        stats["warning"] = "⚠ Please pick up your dumbbell!"
                    else:
                        stats["warning"] = form_warning

                # --- Squats Logic ---
                elif current_exercise == 'squats' and isinstance(exercise_handler, SquatCorrector):
                    # The process_frame method in SquatCorrector already draws landmarks.
                    # We pass `draw_landmarks=False` because we are now handling drawing outside this method.
                    _, rep_count, stage, form_warning = exercise_handler.process_frame(
                        frame, draw_landmarks=False)
                    total = rep_count

                    stats["total"] = total
                    stats["stage"] = stage
                    stats["warning"] = form_warning

                    # Map knee angle to progress bar (if available)
                    if exercise_handler.recording:
                        last_frame_data = exercise_handler.recording[-1]
                        knee_angle = last_frame_data.get('knee_angle')
                        if knee_angle is not None:
                            # Squat progress: 165 (standing) is 0%, 90 (bottom) is 100%
                            stats["progress"] = map_angle_to_progress(
                                knee_angle, min_angle=90, max_angle=165)

                    if total > last_spoken_rep:
                        speak_async(str(total))
                        last_spoken_rep = total
                        if total == round(TARGET_REPS / 2):
                            speak_async("Halfway there, keep it up!")
                        elif total == TARGET_REPS:
                            speak_async("Great set! Take a rest!")

                    # Handle logged errors for squats
                    if exercise_handler.new_error_logged:
                        if hasattr(exercise_handler, 'last_error'):
                            error_side, error_msg = exercise_handler.last_error
                            rep_at_error = total + 1  # Error is for the upcoming rep
                            stats["error_log"].append(
                                f"At Rep {rep_at_error}: {error_msg}")
                            exercise_handler.new_error_logged = False

                # --- Common Logic (Completion, Voice Feedback) ---
                if stats.get("total", 0) >= TARGET_REPS and is_workout_active:
                    is_workout_active = False
                    stats["workout_complete"] = True
                    stats["warning"] = stats["target_hit_message"]
                    speak_async(stats["target_hit_message"])
                    continue

                # Voice feedback for form warnings
                current_time = time.time()
                should_speak = form_warning and (form_warning != last_spoken_feedback or (
                    current_time - last_spoken_time > app.config['APP_CONFIG']['feedback_cooldown_sec']))
                if should_speak:
                    speak_async(form_warning)
                    last_spoken_feedback = form_warning
                    last_spoken_time = current_time
                elif not form_warning and last_spoken_feedback:
                    last_spoken_feedback = ""

            # Encode frame for streaming
            _, buffer = cv2.imencode('.jpg', processed_image)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"⚠️ Frame processing error (continuing): {e}")
            # Don't break; just skip the frame and continue the loop
            continue
        except GeneratorExit:
            # This is triggered when the client disconnects (e.g., closes tab)
            print("Client disconnected. Releasing camera.")
            break

# --- Routes ---


@app.route('/')
def index():
    # On first launch, ensure TARGET_REPS is initialized from config
    global TARGET_REPS
    if 'TARGET_REPS' not in globals():
        TARGET_REPS = app.config['APP_CONFIG'].get('default_target_reps', 15)

    """Redirects the root URL to the exercise selection page."""
    # This makes the exercise selection the default starting page.
    # We will warm up the camera on this page.
    return redirect(url_for('select_exercise'))


@app.route('/dashboard_data')
def dashboard_data():
    """Provides a single endpoint for all dynamic dashboard data."""
    global stats, workout_start_time, is_workout_active

    # 1. Live Workout Summary
    live_stats = stats.copy()
    duration_seconds = 0
    if is_workout_active and workout_start_time:
        duration_seconds = time.time() - workout_start_time

    # Simple calorie estimation: ~0.35 calories per rep
    calories_burned = live_stats.get('total', 0) * 0.35

    # 2. Accuracy Calculation
    total_reps = live_stats.get('total', 0)
    num_errors = len(live_stats.get('error_log', []))
    accuracy = 0
    if total_reps > 0:
        # Ensure accuracy doesn't go below zero
        accuracy = max(0, (total_reps - num_errors) / total_reps) * 100

    return jsonify({
        "live_summary": {
            "reps": live_stats.get('total', 0),
            "sets": 1,  # Assuming 1 set for now
            "duration": round(duration_seconds),
            "calories": round(calories_burned),
            "progress": live_stats.get('progress', 0)
        },
        "accuracy": round(accuracy)
    })


@app.route('/select_exercise')
def select_exercise():
    """Renders the page for selecting an exercise."""

    # Find the last time each exercise was performed
    last_performed = {}
    try:
        cursor.execute("""
            SELECT et.exercise_name, MAX(s.date_time) AS last_time
            FROM sessions s
            JOIN exercise_targets et ON s.exercise_id = et.exercise_id
            GROUP BY et.exercise_name
        """)
        results = cursor.fetchall()
        for row in results:
            last_performed[row['exercise_name'].replace(
                '_', ' ').title()] = row['last_time'].strftime("%Y-%m-%d")
    except Exception as e:
        print("⚠️ Error fetching last performed data:", e)

    # Define available exercises
    available_exercises = [
        {'id': 'bicep_curls', 'name': 'Bicep Curls', 'disabled': False},
        {'id': 'squats', 'name': 'Squats', 'disabled': False},
        {'id': 'push_ups', 'name': 'Push-ups', 'disabled': True},
    ]

    # Add the 'last_performed' date to each exercise
    for exercise in available_exercises:
        exercise['last_performed'] = last_performed.get(
            exercise['name'], 'Never')

    return render_template('dashboard.html', exercises=available_exercises, config=app.config['APP_CONFIG'])


@app.route('/new_dashboard')
def new_dashboard():
    """Renders the new dashboard with charts and history."""
    user_name = app.config['APP_CONFIG'].get('user_name', 'User')
    return render_template('new_dashboard.html', user_name=user_name)


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Renders the settings page and handles form submission."""
    if request.method == 'POST':
        current_config = app.config['APP_CONFIG']
        current_config['user_name'] = request.form.get('user_name', 'User')
        current_config['default_target_reps'] = int(
            request.form.get('default_target_reps', 15))
        current_config['speak_feedback'] = 'speak_feedback' in request.form
        current_config['feedback_cooldown_sec'] = float(
            request.form.get('feedback_cooldown_sec', 5))
        current_config['ask_camera_permission'] = 'ask_camera_permission' in request.form

        save_config(current_config)
        app.config['APP_CONFIG'] = current_config  # Update live config
        global TARGET_REPS
        TARGET_REPS = current_config['default_target_reps']

        return redirect(url_for('settings'))
    return render_template('settings.html', config=load_config())


@app.route('/confirm_camera')
def confirm_camera():
    """Shows a confirmation page before accessing the camera."""
    exercise_id = request.args.get('exercise')
    return render_template('confirm_camera.html', exercise_id=exercise_id)


@app.route('/workout')
def workout():
    global current_exercise, TARGET_REPS
    current_exercise = request.args.get('exercise', 'bicep_curls')
    TARGET_REPS = app.config['APP_CONFIG'].get('default_target_reps', 15)

    # ✅ Open camera only when workout page loads
    get_camera()

    # Pass exercise info to the template
    exercise_name = current_exercise.replace('_', ' ').title()
    intro_video_path = 'Video/Video_Refinement_Request (1).mp4'
    if current_exercise == 'squats':
        intro_video_path = 'Video/Knee_Bending_Exercise_Demonstration.mp4'
    return render_template('workout.html', exercise_name=exercise_name, exercise_id=current_exercise, intro_video=intro_video_path, target_reps=TARGET_REPS)


@app.route('/video_feed')
def video_feed():
    global _camera_active
    cap = get_camera()
    if cap is None:
        print("❌ Could not start video feed — camera unavailable.")
        return Response(status=500)

    _camera_active = True
    print("🎥 /video_feed started — camera active.")
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/history')
def history():
    history_data = []
    unique_exercises = set()

    try:
        # 1. Read from MySQL (unified sessions)
        cursor.execute("""
            SELECT s.date_time, et.exercise_name, s.total_reps, s.duration_seconds, s.avg_angle, s.feedback
            FROM sessions s
            JOIN exercise_targets et ON s.exercise_id = et.exercise_id
            ORDER BY s.date_time DESC
        """)
        db_history = cursor.fetchall()
        for row in db_history:
            exercise_name = row['exercise_name'].replace('_', ' ').title()
            unique_exercises.add(exercise_name)
            history_data.append({
                'timestamp': row['date_time'].strftime("%Y-%m-%d %H:%M:%S"),
                'exercise': exercise_name,
                'reps': row['total_reps'],
                'duration_seconds': row['duration_seconds'],
                'avg_angle': f"{row['avg_angle']:.1f}°" if row.get('avg_angle') else 'N/A',
                'feedback': row.get('feedback', 'N/A')
            })

        # Sort newest first
        history_data.sort(key=lambda x: x['timestamp'], reverse=True)

    except Exception as e:
        print(f"❌ Error fetching history: {e}")

    return render_template('history.html', history=history_data, unique_exercises=list(unique_exercises))


@app.route('/stats')
def get_stats():
    # For bicep curls, we send left/right/total. For others, just total.
    # The frontend should handle this gracefully.
    if current_exercise == 'bicep_curls':
        return jsonify(stats)
    else:
        # Create a compatible structure for other exercises
        response_stats = {
            "total": stats.get("total", 0),
            "stage": stats.get("stage", "-"),
            "warning": stats.get("warning", ""),
            "progress": stats.get("progress", 0),
            "error_log": stats.get("error_log", []),
            "workout_complete": stats.get("workout_complete", False),
            # Add dummy left/right for frontend compatibility if needed
            "left": 0,
            "right": 0
        }
        return jsonify(response_stats)


@app.route('/start', methods=['POST'])
def start_workout():
    global is_workout_active, stats, last_spoken_feedback, last_spoken_time, last_spoken_rep, workout_start_time, current_exercise, exercise_handler, TARGET_REPS
    workout_start_time = time.time()
    is_workout_active = True
    # Reset counters & stats at workout start
    stats = {"left": 0, "right": 0, "total": 0,
             "stage": "-", "warning": "", "progress": 0, "error_log": [],
             "workout_complete": False, "target_hit_message": "Target Hit! Great job! Take a rest."}
    TARGET_REPS = app.config['APP_CONFIG'].get('default_target_reps', 15)
    last_spoken_feedback = ""
    last_spoken_time = 0
    last_spoken_rep = 0

    # Instantiate the correct handler for the selected exercise
    if current_exercise == 'bicep_curls':
        exercise_handler = CurlCounter()
        stats["stage"] = "L- | R-"
    elif current_exercise == 'squats':
        # The app's TTS will be used for rep counting and feedback
        exercise_handler = SquatCorrector(speak_feedback=False)
        stats["stage"] = "standing"
    else:
        exercise_handler = None  # No handler for this exercise yet

    speak_async(f"Let's go for {TARGET_REPS} reps. Starting now.")
    print(
        f"✅ Workout started for {current_exercise}, handler: {type(exercise_handler).__name__}")
    return jsonify({"status": "Workout started"})


def store_session_in_db(exercise_handler, user_id=1, exercise_name='squats'):
    """
    Stores session data in MySQL and generates custom feedback based on
    deviation from target/ideal angles in the 'exercise_targets' table.
    Works for both Squats and Bicep Curls.
    """
    try:
        if exercise_handler is None:
            print("store_session_in_db: exercise_handler is None -> skipping.")
            return {}

        summary = exercise_handler.get_session_summary()
        print(f"store_session_in_db summary for {exercise_name}:", summary)

        # ✅ 1. Fetch exercise details and targets
        form_accuracy = summary.get("form_accuracy", 0)
        num_errors = len(summary.get("rep_issues", []))

        cursor.execute("""
            SELECT exercise_id, ideal_angle, target_angle
            FROM exercise_targets WHERE exercise_name = %s
        """, (exercise_name,))
        row = cursor.fetchone()
        if not row:
            print(f"⚠ No exercise_id found for {exercise_name}")
            return {}

        exercise_id = row['exercise_id']
        ideal_angle = row['ideal_angle']
        target_angle = row['target_angle']

        # ✅ 2. Gather recorded angles (knee or elbow)
        angle_key = 'knee_angle' if exercise_name == 'squats' else 'elbow_angle'
        angles = [r[angle_key] for r in getattr(
            exercise_handler, 'recording', []) if r.get(angle_key) is not None]
        avg_angle = float(np.mean(angles)) if angles else None

        duration_seconds = summary.get('session_time_sec', 0)
        total_reps = summary.get('total_reps', 0)

        # ✅ 3. Fetch previous valid session BEFORE inserting the new one
        cursor.execute("""
            SELECT form_accuracy FROM sessions
            WHERE user_id=%s AND exercise_id=%s 
              AND form_accuracy IS NOT NULL
            ORDER BY date_time DESC LIMIT 1
        """, (user_id, exercise_id))
        last = cursor.fetchone()

        if last:
            last_accuracy = last['form_accuracy']
            print(
                f"📋 Compared with previous session: form_accuracy={last_accuracy}")
        else:
            last_accuracy = None
            print("ℹ️ No previous valid session found — treating this as baseline.")

        # ✅ 4. Compute improvement percent BEFORE saving
        improvement_percent = calculate_bounded_improvement(
            form_accuracy, last_accuracy)
        print(
            f"📊 Improvement Debug: current={form_accuracy}, previous={last_accuracy}, final={improvement_percent}"
        )

        # ✅ 5. Generate Smart Feedback
        feedback = "Good session!"
        if avg_angle:
            diff = avg_angle - target_angle
            diff_percent = (diff / target_angle) * 100

            # For squats (higher angle = shallower depth)
            if exercise_name == 'squats':
                if avg_angle > target_angle + 10:
                    feedback = (f"Your average squat depth is {abs(diff_percent):.1f}% shallower "
                                f"than ideal. Aim for around {target_angle:.1f}° at the bottom.")
                elif avg_angle < target_angle - 10:
                    feedback = (f"Excellent depth! You went {abs(diff_percent):.1f}% deeper "
                                f"than target — great control!")
                else:
                    feedback = (f"Ideal: {target_angle:.1f}°, Your Avg: {avg_angle:.1f}° — "
                                "Excellent form!")

            # For curls (lower angle = better flexion)
            elif exercise_name == 'bicep_curls':
                if avg_angle > target_angle + 10:
                    feedback = (f"Try curling higher! Ideal bottom angle: {target_angle:.1f}°, "
                                f"Your Avg: {avg_angle:.1f}° ({abs(diff_percent):.1f}% above ideal).")
                elif avg_angle < target_angle - 10:
                    feedback = (
                        f"Perfect contraction! You’re {abs(diff_percent):.1f}% beyond target range — keep it up!")
                else:
                    feedback = (f"Ideal: {target_angle:.1f}°, Your Avg: {avg_angle:.1f}° — "
                                "Good range of motion!")

        # ✅ 6. Save session summary to DB
        cursor.execute("""
            INSERT INTO sessions (user_id, exercise_id, date_time, total_reps, avg_angle,
                                  improvement_percent, feedback, duration_seconds, form_accuracy)
            VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s, %s)
        """, (user_id, exercise_id, total_reps, avg_angle,
              improvement_percent, feedback, duration_seconds, form_accuracy))
        db.commit()
        session_id = cursor.lastrowid

        # ✅ 7. Store per-rep data (angle history)
        rep_count = 0
        for i, r in enumerate(getattr(exercise_handler, 'recording', []), start=1):
            if r.get(angle_key):
                cursor.execute("""
                    INSERT INTO rep_data (session_id, rep_number, angle, timestamp)
                    VALUES (%s, %s, %s, NOW())
                """, (session_id, i, r[angle_key]))
                rep_count += 1
        db.commit()

        print(f"✅ Stored {exercise_name} session ID {session_id}: "
              f"{total_reps} reps, avg {avg_angle}°, feedback: {feedback}")

        return {
            "session_id": session_id,
            "avg_angle": avg_angle,
            "improvement_percent": improvement_percent,
            "feedback": feedback,
            "total_reps": total_reps,
            "target_angle": target_angle,
            "accuracy": round(form_accuracy, 1),
            "num_errors": num_errors
        }

    except Exception as e:
        print("❌ DB Error in store_session_in_db:", e)
        db.rollback()
        return {}


@app.route('/stop', methods=['POST'])
def stop_exercise():
    global is_workout_active, workout_start_time, stats, current_exercise, exercise_handler
    # Capture the exercise name before it's cleared
    exercise_name_at_stop = current_exercise
    is_workout_active = False

    # 🔄 Safety auto-recover handler if lost before stop
    if exercise_handler is None and exercise_name_at_stop:
        print("⚠️ Handler missing, attempting recovery...")
        if exercise_name_at_stop == 'bicep_curls':
            exercise_handler = CurlCounter()
        elif exercise_name_at_stop == 'squats':
            exercise_handler = SquatCorrector(speak_feedback=False)

    session_info = {}

    # Debug: show current state
    print("STOP called. current_exercise:", current_exercise)
    print("exercise_handler is None?", exercise_handler is None)

    # Save time & stats (computed even if handler is None)
    if workout_start_time:
        duration = time.time() - workout_start_time
        total_reps = stats.get('total', 0)
        exercise_name = exercise_name_at_stop.replace(
            '_', ' ').title() if current_exercise else "Unknown Exercise"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ✅ NEW DEBUG LOG: show if we will attempt DB save
        print(
            f"Attempting to save session: exercise={exercise_name}, reps={total_reps}, duration={round(duration)}s")

        if exercise_handler:
            print(f"Analyzing and saving {exercise_name_at_stop} session...")
            session_info = store_session_in_db(
                exercise_handler, user_id=1, exercise_name=exercise_name_at_stop)
        else:
            print("No valid exercise handler, skipping DB save.")

        # Optional CSV backup (disabled by default)
        if ENABLE_CSV_BACKUP:
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            file_exists = os.path.isfile(HISTORY_FILE)
            with open(HISTORY_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(
                        ['timestamp', 'exercise', 'reps', 'duration_seconds'])
                writer.writerow([timestamp, exercise_name_at_stop.replace('_', ' ').title(),
                                total_reps, round(duration)])

        workout_start_time = None  # Reset start time

    # ---- After workout is saved ----
    # Initialize summary to an empty dict for safety
    summary_from_handler = {}
    if exercise_handler:
        try:
            # This assumes your handler has a method like get_session_summary()
            summary_from_handler = exercise_handler.get_session_summary()
            # 🧮 Ensure form accuracy is captured for SquatCorrector
            if "form_accuracy" not in summary_from_handler:
                if hasattr(exercise_handler, "get_session_summary"):
                    try:
                        # Re-call to ensure all metrics are present
                        summary_from_handler = exercise_handler.get_session_summary()
                    except Exception as e:
                        print(f"⚠️ Could not re-compute form_accuracy: {e}")
                        summary_from_handler["form_accuracy"] = 0.0

            # ✅ Fallback to prevent all-zero summaries if motion was detected
            if summary_from_handler.get("total_reps", 0) == 0:
                if len(getattr(exercise_handler, "recording", [])) > 30:
                    summary_from_handler["total_reps"] = 1
                    summary_from_handler["form_accuracy"] = 50.0
            print("stop_exercise: summary:", summary_from_handler)
        except Exception as e:
            print(f"⚠️ Error getting session summary from handler: {e}")
            # summary_from_handler remains {}

    if exercise_handler:  # Still need this check for cleanup and specific return status
        # Clean up
        if hasattr(exercise_handler, 'release'):
            exercise_handler.release()
        exercise_handler = None
        current_exercise = None

        # Send summary to frontend
        # ---- build a dynamic explanation for frontend ----
        explanation_text = ""
        try:
            avg_angle = session_info.get("avg_angle")
            target_angle = session_info.get("target_angle")
            feedback_text = session_info.get(
                "feedback", "") or summary_from_handler.get("feedback", "")
            exercise_name = exercise_name_at_stop.replace('_', ' ').title()
            key_name = exercise_name.lower()
            if avg_angle is not None and target_angle is not None:
                diff = avg_angle - target_angle
                diff_percent = (diff / target_angle) * \
                    100 if target_angle else 0
                if "squat" in key_name:
                    if diff_percent > 10:
                        explanation_text = (f"Your average squat depth is {abs(diff_percent):.1f}% shallower than ideal. "
                                            f"Aim for around {target_angle:.1f}° at the bottom.")
                    elif diff_percent < -10:
                        explanation_text = (
                            f"Excellent depth! You went {abs(diff_percent):.1f}% deeper than target — great control!")
                    else:
                        explanation_text = (
                            f"Ideal: {target_angle:.1f}°, Your Avg: {avg_angle:.1f}° — Excellent form!")
                elif "bicep" in key_name or "curl" in key_name:
                    if diff_percent > 10:
                        explanation_text = (f"Try curling higher! Ideal bottom angle: {target_angle:.1f}°, "
                                            f"Your Avg: {avg_angle:.1f}° ({abs(diff_percent):.1f}% above ideal).")
                    elif diff_percent < -10:
                        explanation_text = (
                            f"Perfect contraction! You’re {abs(diff_percent):.1f}% beyond target range — keep it up!")
                    else:
                        explanation_text = (
                            f"Ideal: {target_angle:.1f}°, Your Avg: {avg_angle:.1f}° — Good range of motion!")
                else:
                    # generic fallback
                    explanation_text = feedback_text or "Good session!"
            else:
                explanation_text = feedback_text or "Good session!"
        except Exception as e:
            print("⚠️ Error building explanation_text:", e)
            explanation_text = summary_from_handler.get(
                "feedback", session_info.get("feedback", "Good session!"))

        # Merge handler summary with DB-saved info for the most complete picture
        merged_accuracy = summary_from_handler.get("form_accuracy", None)
        if (not merged_accuracy or merged_accuracy == 0.0) and "accuracy" in session_info:
            merged_accuracy = session_info["accuracy"]

        return jsonify({
            "status": "stopped",
            "message": "Session saved successfully!",
            "summary": {
                "exercise": summary_from_handler.get("exercise", exercise_name_at_stop),
                "total_reps": summary_from_handler.get("total_reps", session_info.get("total_reps", 0)),
                "avg_angle": session_info.get("avg_angle", summary_from_handler.get("avg_angle", 0)),
                # ✅ FIX: improvement should only come from DB summary (session_info)
                "improvement_percent": session_info.get("improvement_percent", 0.0),
                "feedback": session_info.get("feedback", summary_from_handler.get("feedback", "Good session!")),
                "explanation": explanation_text,
                "form_accuracy": round(
                    summary_from_handler.get(
                        "form_accuracy", session_info.get("accuracy", 0.0)), 1
                ),
                "smoothness_score": summary_from_handler.get("smoothness_score"),
                "issue_counts": summary_from_handler.get("issue_counts", {})
            }
        })

    else:
        print("exercise_handler is None, returning fallback summary.")

    # Even if handler is None, provide a default/fallback summary for the popup
    fallback_summary = {
        "exercise": current_exercise.replace('_', ' ').title() if current_exercise else "Unknown",
        "total_reps": stats.get("total", 0),
        "avg_angle": 0,
        "improvement_percent": 0,
        "feedback": "Workout stopped successfully.",
        "explanation": "Session ended early or no data recorded. Try again to capture full analysis.",
        "form_accuracy": 0.0,
        "smoothness_score": None,
        "issue_counts": {}
    }

    # Clean reset to ensure fresh session next time
    exercise_handler = None
    current_exercise = None
    is_workout_active = False

    return jsonify({
        "status": "stopped",
        "message": "Workout stopped (fallback summary).",
        "summary": fallback_summary
    })


@app.route('/set_target_reps', methods=['POST'])
def set_target_reps():
    global TARGET_REPS, last_spoken_rep
    data = request.get_json()
    new_target = data.get('target')
    if new_target and int(new_target) > 0:
        TARGET_REPS = int(new_target)
        last_spoken_rep = 0  # Reset rep count to avoid confusion
    return jsonify({"status": "Target updated", "new_target": TARGET_REPS})


@app.route('/toggle_landmarks', methods=['POST'])
def toggle_landmarks():
    """Sets the server-side state for showing/hiding landmarks."""
    global show_landmarks
    data = request.get_json()
    show_landmarks = data.get('show', True)
    print(f"Landmark visibility set to: {show_landmarks}")
    return jsonify({"status": "success", "show_landmarks": show_landmarks})


@app.route('/release_camera', methods=['POST'])
def release_camera_route():
    """Explicitly release the global camera when leaving the workout page."""
    try:
        release_camera()
        print("✅ Camera released manually via /release_camera.")
        return jsonify({"status": "success", "message": "Camera released."})
    except Exception as e:
        print("⚠️ Error releasing camera:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/delete_entry', methods=['POST'])
def delete_entry():
    try:
        data = request.get_json()
        timestamp_to_delete = data.get('timestamp')

        if not timestamp_to_delete:
            return jsonify({"status": "error", "message": "No timestamp provided"}), 400

        print(f"🕒 Attempting to delete entry at: {timestamp_to_delete}")

        # Use flexible timestamp comparison (±1 sec)
        cursor.execute("""
            SELECT session_id FROM sessions 
            WHERE ABS(TIMESTAMPDIFF(SECOND, date_time, %s)) <= 1
        """, (timestamp_to_delete,))
        session = cursor.fetchone()

        if not session:
            print(f"⚠ No session found for timestamp: {timestamp_to_delete}")
            return jsonify({"status": "error", "message": f"No matching session found for {timestamp_to_delete}"}), 404

        session_id = session['session_id']

        # Delete from child tables first (rep_data → sessions)
        cursor.execute(
            "DELETE FROM rep_data WHERE session_id = %s", (session_id,))
        cursor.execute(
            "DELETE FROM sessions WHERE session_id = %s", (session_id,))
        db.commit()

        print(f"🗑 Deleted session ID {session_id} successfully.")
        return jsonify({"status": "success", "message": "Entry deleted from database"})

    except Exception as e:
        print("❌ Error deleting entry:", e)
        db.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
