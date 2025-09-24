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


# --- Thread-safe Global State & Processing ---
frame_lock = threading.Lock()
latest_frame = None  # This will hold the latest processed frame (as bytes)
is_processing_thread_running = True


# --- Initialize state ---
cap = cv2.VideoCapture(0)
detector = PoseDetector()
counter = CurlCounter()
yolo_model = YOLO("models/best.pt")

is_workout_active = False

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


# --- Frame Generator ---


def generate_frames():
    global stats, is_workout_active, last_spoken_feedback, last_spoken_time, last_spoken_rep
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not is_workout_active:
            # Show raw paused frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue

        # --- YOLO Dumbbell Detection ---
        yolo_results = yolo_model(frame, verbose=False)
        dumbbell_detected = any(len(r.boxes) > 0 for r in yolo_results)

        form_warning = ""  # For form feedback

        # --- Pose Detection ---
        image, results = detector.detect(frame)
        if results.pose_landmarks:
            counter.process(image, results.pose_landmarks.landmark)
            left, right = counter.left_counter, counter.right_counter
            total = min(left, right)
            stats["left"], stats["right"], stats["total"] = left, right, total
            stats["stage"] = f"L-{counter.left_stage or '-'} | R-{counter.right_stage or '-'}"

            # Check for form feedback from the counter
            feedbacks = [f for f in [
                counter.left_feedback, counter.right_feedback] if f]
            if feedbacks:
                form_warning = " | ".join(sorted(feedbacks))

            # --- Interactive Voice Trainer Logic ---
            # Log errors with rep context
            if counter.new_error_logged:
                error_side, error_msg = counter.last_error
                # The rep number when the error occurred
                rep_at_error = (stats[error_side] + 1)
                stats["error_log"].append(
                    f"At Rep {rep_at_error} ({error_side.capitalize()}): {error_msg}")
                counter.new_error_logged = False  # Reset flag

            # Use the higher of the two rep counts as the primary
            current_rep_count = max(left, right)
            if current_rep_count > last_spoken_rep:
                speak_async(str(current_rep_count))
                last_spoken_rep = current_rep_count

                # Give mid-set encouragement
                if current_rep_count == round(TARGET_REPS / 2):
                    speak_async("Halfway there, keep it up!")
                elif current_rep_count == TARGET_REPS:
                    speak_async("Great set! Take a rest.")

            # --- Target Reps Hit Logic ---
            if total >= TARGET_REPS and is_workout_active:
                is_workout_active = False  # Stop the workout
                stats["workout_complete"] = True
                stats["warning"] = stats["target_hit_message"]
                speak_async(stats["target_hit_message"])
                continue
                # Continue to next loop iteration to immediately show paused frame

            # Update progress
            # We'll use the right arm for the progress bar for simplicity
            stats["progress"] = map_angle_to_progress(counter.right_angle)

        # Set the final warning message, prioritizing dumbbell detection
        # Do not overwrite the "workout_complete" message
        if stats.get("workout_complete"):
            pass
        elif not dumbbell_detected:
            stats["warning"] = "⚠ Please pick up your dumbbell!"
        else:
            stats["warning"] = form_warning
            # --- Voice Feedback Logic ---
            current_time = time.time()
            # Speak if there's a new warning, or if the same warning persists after the cooldown
            should_speak = form_warning and (form_warning != last_spoken_feedback or (
                current_time - last_spoken_time > FEEDBACK_COOLDOWN))
            if should_speak:
                speak_async(form_warning)
                last_spoken_feedback = form_warning
                last_spoken_time = current_time
            elif not form_warning and last_spoken_feedback:
                last_spoken_feedback = ""  # Reset when form is good

        # detector.draw_landmarks(image, results) # Removed to keep the feed clean

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# --- Routes ---
@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/workout')
def workout():
    # This was the old index.html
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
    global is_workout_active, counter, stats, last_spoken_feedback, last_spoken_time, last_spoken_rep, workout_start_time
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
    speak_async(f"Let's go for {TARGET_REPS} reps. Starting now.")
    return jsonify({"status": "Workout started"})


@app.route('/stop', methods=['POST'])
def stop_workout():
    global is_workout_active, workout_start_time, stats
    is_workout_active = False

    if workout_start_time:
        duration = time.time() - workout_start_time
        total_reps = stats.get('total', 0)
        exercise_name = "Bicep Curls"
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
# --- Cleanup ---


@atexit.register
def cleanup():
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(debug=True)
