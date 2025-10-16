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
import csv
import os

app = Flask(__name__)

# --- TTS Engine Setup ---
tts_engine = pyttsx3.init()


def speak_async(text):
    """Run TTS in a separate thread to avoid blocking the main app."""
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
            _camera_instance = cv2.VideoCapture(0)
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
FEEDBACK_COOLDOWN = 5  # seconds
TARGET_REPS = 15
last_spoken_rep = 0
workout_start_time = None

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

            if is_workout_active:
                # --- Bicep Curls Logic ---
                if current_exercise == 'bicep_curls' and isinstance(exercise_handler, CurlCounter):
                    yolo_results = yolo_model(processed_image, verbose=False)
                    dumbbell_detected = any(
                        len(r.boxes) > 0 for r in yolo_results)
                    form_warning = ""

                    processed_image, results = detector.detect(processed_image)
                    landmarks_visible = results.pose_landmarks and are_landmarks_visible(
                        results.pose_landmarks)

                    if landmarks_visible:
                        exercise_handler.process(
                            processed_image, results.pose_landmarks.landmark)
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

                        stats["progress"] = map_angle_to_progress(
                            exercise_handler.right_angle)

                    elif results.pose_landmarks:
                        form_warning = "Please make sure your full upper body is visible."

                    if not dumbbell_detected:
                        stats["warning"] = "⚠ Please pick up your dumbbell!"
                    else:
                        stats["warning"] = form_warning

                # --- Squats Logic ---
                elif current_exercise == 'squats' and isinstance(exercise_handler, SquatCorrector):
                    processed_image, rep_count, stage, form_warning = exercise_handler.process_frame(
                        processed_image)
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
                    current_time - last_spoken_time > FEEDBACK_COOLDOWN))
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

    return render_template('dashboard.html', exercises=available_exercises)


@app.route('/new_dashboard')
def new_dashboard():
    """Renders the new dashboard with charts and history."""
    # We no longer pass history data to the main dashboard.
    return render_template('new_dashboard.html')


@app.route('/workout')
def workout():
    global current_exercise
    current_exercise = request.args.get('exercise', 'bicep_curls')
    # Pass exercise info to the template
    exercise_name = current_exercise.replace('_', ' ').title()
    return render_template('workout.html', exercise_name=exercise_name, exercise_id=current_exercise)


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
