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
import atexit
import json
import csv
import os

app = Flask(__name__)
# Necessary for session management
app.secret_key = 'your_very_secret_key_for_sessions'

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


def speak_async(text):
    """Run TTS in a separate thread to avoid blocking the main app."""
    if not app.config['APP_CONFIG'].get('speak_feedback', True):
        return

    def run():
        tts_engine.say(text)
        tts_engine.runAndWait()
    thread = threading.Thread(target=run)
    thread.start()


# --- Camera Management ---
_camera_instance = None
_camera_lock = threading.Lock()


def get_camera():
    """Initializes or returns the global camera instance."""
    global _camera_instance
    with _camera_lock:
        if _camera_instance is None:
            camera_index = app.config['APP_CONFIG'].get('camera_index', 0)
            _camera_instance = cv2.VideoCapture(camera_index)
            if not _camera_instance.isOpened():
                print("Error: Could not open video stream.")
                _camera_instance = None  # Ensure it's None if opening fails
        return _camera_instance


def release_camera():
    """Releases the global camera instance."""
    global _camera_instance
    with _camera_lock:
        if _camera_instance is not None:
            _camera_instance.release()
            _camera_instance = None
            print("Camera released.")


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

# --- History File ---
HISTORY_FILE = 'data/workout_history.csv'

# --- Landmark Visibility Check ---


def are_landmarks_visible(landmarks, visibility_threshold=0.7):
    """Checks if essential landmarks for bicep curls are visible."""
    # Define the essential landmarks for bicep curls
    required_landmarks = [
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
        mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
        mp.solutions.pose.PoseLandmark.LEFT_WRIST,
        mp.solutions.pose.PoseLandmark.LEFT_HIP,
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
        mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
        mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    ]
    for landmark_index in required_landmarks:
        if landmarks.landmark[landmark_index.value].visibility < visibility_threshold:
            return False
    return True

# --- Frame Generator ---


def generate_frames():
    global stats, is_workout_active, last_spoken_feedback, last_spoken_time, last_spoken_rep, exercise_handler, current_exercise

    cap = get_camera()
    if cap is None:
        # Placeholder for camera error
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "CAMERA ERROR", (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', blank_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            # Always display the live camera feed.
            # 'processed_image' will hold the frame that gets displayed.
            processed_image = frame.copy()

            # --- Always-on Pose Detection and Landmark Drawing ---
            # Detect pose first on the raw frame.
            # The detector returns the image it processed, which we can draw on.
            processed_image, results = detector.detect(processed_image)
            landmarks_visible = results.pose_landmarks and are_landmarks_visible(
                results.pose_landmarks)

            # Draw landmarks if the toggle is on, regardless of workout state.
            if show_landmarks:
                detector.draw_landmarks(processed_image, results)

            if is_workout_active:
                # --- Bicep Curls Logic ---
                if current_exercise == 'bicep_curls' and isinstance(exercise_handler, CurlCounter):
                    yolo_results = yolo_model(processed_image, verbose=False)

                    if landmarks_visible:
                        exercise_handler.process(
                            processed_image, results.pose_landmarks.landmark)  # This method doesn't modify the image
                        left, right = exercise_handler.left_counter, exercise_handler.right_counter
                        total = min(left, right)
                        stats["left"], stats["right"], stats["total"] = left, right, total
                        stats["stage"] = f"L-{exercise_handler.left_stage or '-'} | R-{exercise_handler.right_stage or '-'}"

                        feedbacks = [f for f in [
                            exercise_handler.left_feedback, exercise_handler.right_feedback] if f]
                        if feedbacks:
                            form_warning = " | ".join(sorted(feedbacks))

                        if exercise_handler.new_error_logged:
                            error_side, error_msg = exercise_handler.last_error
                            rep_at_error = (stats[error_side] + 1)
                            stats["error_log"].append(
                                f"At Rep {rep_at_error} ({error_side.capitalize()}): {error_msg}")
                            exercise_handler.new_error_logged = False

                        current_rep_count = max(left, right)
                        if current_rep_count > last_spoken_rep:
                            speak_async(str(current_rep_count))
                            last_spoken_rep = current_rep_count

                            if current_rep_count == round(TARGET_REPS / 2):
                                speak_async("Halfway there, keep it up!")
                            elif current_rep_count == TARGET_REPS:
                                speak_async("Great set! Take a rest!")

                        # Update progress based on the active arm
                        if exercise_handler.left_stage == 'up':
                            stats["progress"] = map_angle_to_progress(
                                exercise_handler.left_angle)
                        elif exercise_handler.right_stage == 'up':
                            stats["progress"] = map_angle_to_progress(
                                exercise_handler.right_angle)
                        else:
                            # If both arms are down, progress is 0
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
                        error_msg = exercise_handler.last_error_message
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
            frame_to_send = cv2.flip(processed_image, 1)
            _, buffer = cv2.imencode('.jpg', frame_to_send)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        release_camera()  # Ensure camera is released when the generator exits


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
    # --- Camera Warm-up ---
    # Start initializing the camera in the background as soon as the user
    # lands on this page. This prevents the delay when the workout starts.
    threading.Thread(target=get_camera, daemon=True).start()

    # Define available exercises
    available_exercises = [
        {'id': 'bicep_curls', 'name': 'Bicep Curls', 'disabled': False},
        {'id': 'squats', 'name': 'Squats', 'disabled': False},
        {'id': 'push_ups', 'name': 'Push-ups', 'disabled': True},
    ]

    # Find the last time each exercise was performed
    last_performed = {}
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', newline='') as f:
            # Read history into a list to sort it
            reader = csv.DictReader(f)
            history_list = sorted(list(reader), key=lambda x: x.get(
                'timestamp', ''), reverse=True)

            for entry in history_list:
                exercise_name = entry.get('exercise')
                if exercise_name and exercise_name not in last_performed:
                    last_performed[exercise_name] = entry.get(
                        'timestamp', '').split(' ')[0]

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
    # Pass exercise info to the template
    exercise_name = current_exercise.replace('_', ' ').title()
    intro_video_path = 'Video/Video_Refinement_Request (1).mp4'
    if current_exercise == 'squats':
        intro_video_path = 'Video/Knee_Bending_Exercise_Demonstration.mp4'
    return render_template('workout.html', exercise_name=exercise_name, exercise_id=current_exercise, intro_video=intro_video_path, target_reps=TARGET_REPS)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/history')
def history():
    history_data = []
    unique_exercises = set()
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            history_list = list(reader)
            # Reverse the list to show the most recent workouts first
            history_data = sorted(
                history_list, key=lambda x: x['timestamp'], reverse=True)
            for entry in history_list:
                if entry.get('exercise'):
                    unique_exercises.add(entry['exercise'])

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
    global is_workout_active, stats, last_spoken_feedback, last_spoken_time, last_spoken_rep, workout_start_time, current_exercise, exercise_handler
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
    return jsonify({"status": "Workout started"})


@app.route('/stop', methods=['POST'])
def stop_workout():
    global is_workout_active, workout_start_time, stats, current_exercise, exercise_handler
    is_workout_active = False

    # If there's a handler with a release method, call it
    if hasattr(exercise_handler, 'release'):
        exercise_handler.release()
    exercise_handler = None

    if workout_start_time:
        duration = time.time() - workout_start_time
        total_reps = stats.get('total', 0)
        exercise_name = current_exercise.replace(
            '_', ' ').title() if current_exercise else "Unknown Exercise"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ensure data directory exists
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

        # Write to CSV
        file_exists = os.path.isfile(HISTORY_FILE)
        with open(HISTORY_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            # Write header if file is new
            if not file_exists:
                writer.writerow(
                    ['timestamp', 'exercise', 'reps', 'duration_seconds'])
            # Write workout data
            writer.writerow([timestamp, exercise_name,
                            total_reps, round(duration)])

        workout_start_time = None  # Reset start time

    return jsonify({"status": "Workout stopped"})


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


@app.route('/delete_entry', methods=['POST'])
def delete_entry():
    data = request.get_json()
    timestamp_to_delete = data.get('timestamp')

    if not timestamp_to_delete or not os.path.exists(HISTORY_FILE):
        return jsonify({"status": "error", "message": "Invalid request or file not found"}), 400

    # Read all entries except the one to be deleted
    updated_history = []
    with open(HISTORY_FILE, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row['timestamp'] != timestamp_to_delete:
                updated_history.append(row)

    # Rewrite the CSV file with the updated history
    with open(HISTORY_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_history)

    return jsonify({"status": "success", "message": "Entry deleted"})


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
