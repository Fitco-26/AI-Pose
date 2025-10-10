import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import cv2
from ultralytics import YOLO
from modules.pose_detector import PoseDetector
from modules.curl_counter import CurlCounter
from modules.posture.utils.angle_utils import map_angle_to_progress
import pyttsx3
import threading
import time
from datetime import datetime
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
yolo_model = YOLO("models/best.pt")
is_workout_active = False
current_exercise = None  # To store the selected exercise

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

# --- Frame Generator ---


def generate_frames():
    global stats, is_workout_active, last_spoken_feedback, last_spoken_time, last_spoken_rep

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
                # --- Processing (YOLO, Pose Detection, Counter) ---
                yolo_results = yolo_model(processed_image, verbose=False)
                dumbbell_detected = any(len(r.boxes) > 0 for r in yolo_results)

                form_warning = ""  # For form feedback

                # --- Pose Detection ---
                processed_image, results = detector.detect(processed_image)
                if results.pose_landmarks:
                    counter.process(processed_image,
                                    results.pose_landmarks.landmark)
                    left, right = counter.left_counter, counter.right_counter
                    total = min(left, right)
                    stats["left"], stats["right"], stats["total"] = left, right, total
                    stats["stage"] = f"L-{counter.left_stage or '-'} | R-{counter.right_stage or '-'}"

                    feedbacks = [f for f in [
                        counter.left_feedback, counter.right_feedback] if f]
                    if feedbacks:
                        form_warning = " | ".join(sorted(feedbacks))

                    if counter.new_error_logged:
                        error_side, error_msg = counter.last_error
                        rep_at_error = (stats[error_side] + 1)
                        stats["error_log"].append(
                            f"At Rep {rep_at_error} ({error_side.capitalize()}): {error_msg}")
                        counter.new_error_logged = False

                    current_rep_count = max(left, right)
                    if current_rep_count > last_spoken_rep:
                        speak_async(str(current_rep_count))
                        last_spoken_rep = current_rep_count

                        if current_rep_count == round(TARGET_REPS / 2):
                            speak_async("Halfway there, keep it up!")
                        elif current_rep_count == TARGET_REPS:
                            speak_async("Great set! Take a rest!")

                    if total >= TARGET_REPS and is_workout_active:
                        is_workout_active = False
                        stats["workout_complete"] = True
                        stats["warning"] = stats["target_hit_message"]
                        # Speak the completion message
                        speak_async(stats["target_hit_message"])
                        continue

                    stats["progress"] = map_angle_to_progress(
                        counter.right_angle)

                if stats.get("workout_complete"):
                    pass
                elif not dumbbell_detected:
                    stats["warning"] = "⚠ Please pick up your dumbbell!"
                else:
                    stats["warning"] = form_warning
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
def dashboard():
    # The new dashboard will be the main page.
    # We pass the history data directly on initial load.
    history_data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            history_data = sorted(
                list(reader), key=lambda x: x['timestamp'], reverse=True)
    return render_template('new_dashboard.html', history=history_data)


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
    # Define available exercises
    available_exercises = [
        {'id': 'bicep_curls', 'name': 'Bicep Curls', 'disabled': False},
        {'id': 'squats', 'name': 'Squats', 'disabled': True},
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


@app.route('/workout')
def workout():
    global current_exercise
    current_exercise = request.args.get('exercise', 'bicep_curls')
    return render_template('workout.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/history')
def history():
    history_data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            # Reverse the list to show the most recent workouts first
            history_data = sorted(
                list(reader), key=lambda x: x['timestamp'], reverse=True)

    return render_template('history.html', history=history_data)


@app.route('/stats')
def get_stats():
    return jsonify(stats)


@app.route('/start', methods=['POST'])
def start_workout():
    global is_workout_active, counter, stats, last_spoken_feedback, last_spoken_time, last_spoken_rep, workout_start_time, current_exercise
    workout_start_time = time.time()
    is_workout_active = True
    # Reset counters & stats at workout start
    counter = CurlCounter()
    stats = {"left": 0, "right": 0, "total": 0,
             "stage": "L- | R-", "warning": "", "progress": 0, "error_log": [],
             "workout_complete": False, "target_hit_message": "Target Hit! Great job! Take a rest."}
    last_spoken_feedback = ""
    last_spoken_time = 0
    last_spoken_rep = 0

    # Here you can add logic for different counters based on `current_exercise`
    # For now, we only have CurlCounter
    if current_exercise == 'bicep_curls':
        counter = CurlCounter()

    speak_async(f"Let's go for {TARGET_REPS} reps. Starting now.")
    return jsonify({"status": "Workout started"})


@app.route('/stop', methods=['POST'])
def stop_workout():
    global is_workout_active, workout_start_time, stats, current_exercise
    is_workout_active = False

    if workout_start_time:
        duration = time.time() - workout_start_time
        total_reps = stats.get('total', 0)
        exercise_name = current_exercise.replace(
            '_', ' ').title() if current_exercise else "Bicep Curls"
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


@app.route('/chart_data')
def chart_data():
    """Processes workout history to provide data for charts."""
    weekly_progress = {
        "labels": [],
        "data": []
    }

    if os.path.exists(HISTORY_FILE):
        # --- Weekly Progress Chart (Last 7 Days) ---
        today = datetime.now()
        last_7_days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]

        # Initialize labels and a dictionary for daily totals
        weekly_progress["labels"] = [day.strftime("%a") for day in last_7_days]
        daily_totals = {day.strftime("%Y-%m-%d"): 0 for day in last_7_days}

        with open(HISTORY_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    workout_date_str = row['timestamp'].split(' ')[0]
                    # Ensure the date is within the last 7 days
                    workout_datetime = datetime.strptime(
                        workout_date_str, "%Y-%m-%d")
                    if workout_datetime.date() in [d.date() for d in last_7_days]:
                        duration_minutes = float(row['duration_seconds']) / 60
                        daily_totals[workout_date_str] += duration_minutes
                except (ValueError, KeyError):
                    continue  # Skip rows with bad data

        weekly_progress["data"] = [
            round(daily_totals[day.strftime("%Y-%m-%d")]) for day in last_7_days]

    return jsonify({"weekly_progress": weekly_progress})


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
