import cv2
import os
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QPushButton, QProgressBar, QSplitter
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
from modules.curl_counter import CurlCounter
from modules.pose_detector import PoseDetector


class WorkoutScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # -------- Video Feed --------
        self.video_label = QLabel("Camera loading...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        # double click to toggle fullscreen
        self.video_label.mouseDoubleClickEvent = self.toggle_fullscreen

        # -------- Stats Panel --------
        self.rep_label = QLabel("Left: 0 | Right: 0 | Total: 0")
        self.rep_label.setStyleSheet(
            "font-size: 22px; color: #00ff00; font-weight: bold;"
        )

        self.stage_label = QLabel("Stage: - | -")
        self.stage_label.setStyleSheet("font-size: 18px; color: yellow;")

        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("font-size: 18px; color: red;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            "QProgressBar::chunk { background-color: #00bfff; }"
        )

        self.start_button = QPushButton("Start Workout")
        self.stop_button = QPushButton("Stop Workout")

        stats_layout = QVBoxLayout()
        stats_layout.addWidget(self.rep_label)
        stats_layout.addWidget(self.stage_label)
        stats_layout.addWidget(self.warning_label)
        stats_layout.addWidget(self.progress_bar)
        stats_layout.addWidget(self.start_button)
        stats_layout.addWidget(self.stop_button)

        self.stats_panel = QWidget()
        self.stats_panel.setLayout(stats_layout)

        # -------- Split Layout --------
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.video_label)
        self.splitter.addWidget(self.stats_panel)

        # Default ratio: 70% camera, 30% stats
        self.splitter.setSizes([700, 300])

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.splitter)
        self.setLayout(main_layout)

        # -------- State & Models --------
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.detector = PoseDetector()
        self.counter = CurlCounter()
        self.yolo_model = YOLO("models/best.pt")

        # -------- Button Actions --------
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)

        # -------- Fullscreen Toggle State --------
        self.fullscreen = False

        # -------- Window Setup --------
        self.setWindowTitle("Workout Screen")

        # -------- Load Stylesheet --------
        qss_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "styles", "styles.qss"
        )
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.setText("Workout stopped.")

    def update_frame(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # -------- YOLO Dumbbell Detection --------
        yolo_results = self.yolo_model(frame)
        dumbbell_detected = any(len(r.boxes) > 0 for r in yolo_results)

        for result in yolo_results:
            frame = result.plot()

        self.warning_label.setText(
            "" if dumbbell_detected else "⚠ Please pick up your dumbbell!"
        )

        # -------- Pose Detection --------
        image, results = self.detector.detect(frame)
        if results.pose_landmarks:
            # Update curl counter
            self.counter.process(image, results.pose_landmarks.landmark)

            left = self.counter.left_counter
            right = self.counter.right_counter
            total = left + right
            self.rep_label.setText(
                f"Left: {left} | Right: {right} | Total: {total}"
            )

            left_stage = self.counter.left_stage if self.counter.left_stage else "-"
            right_stage = self.counter.right_stage if self.counter.right_stage else "-"
            self.stage_label.setText(
                f"Stage: L-{left_stage} | R-{right_stage}"
            )

            # Calculate right arm angle → update progress bar
            angle = self._calculate_angle(results.pose_landmarks.landmark)
            progress = self._map_angle_to_progress(angle)
            self.progress_bar.setValue(progress)

        self.detector.draw_landmarks(image, results)

        # -------- Display in QLabel --------
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def _calculate_angle(self, landmarks):
        """Calculate elbow angle using Mediapipe landmarks (right arm)."""
        shoulder = np.array([landmarks[12].x, landmarks[12].y])
        elbow = np.array([landmarks[14].x, landmarks[14].y])
        wrist = np.array([landmarks[16].x, landmarks[16].y])

        v1 = shoulder - elbow
        v2 = wrist - elbow

        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(cosine))
        return angle

    def _map_angle_to_progress(self, angle, min_angle=30, max_angle=160):
        """Map elbow angle (30–160°) to progress bar (100–0%)."""
        angle = max(min_angle, min(max_angle, angle))  # clamp
        progress = 100 - int((angle - min_angle) * 100 /
                             (max_angle - min_angle))
        return progress

    def toggle_fullscreen(self, event):
        """Toggle fullscreen camera view by double-clicking on video feed."""
        if self.fullscreen:
            # Restore 70/30 layout
            self.stats_panel.show()
            self.splitter.setSizes([700, 300])
            self.fullscreen = False
        else:
            # Hide stats panel → 100% video feed
            self.stats_panel.hide()
            self.splitter.setSizes([1000, 0])
            self.fullscreen = True


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    # ✅ Create the WorkoutScreen as a top-level window
    window = WorkoutScreen()
    # ensures title bar + minimize/maximize/close
    window.setWindowFlags(Qt.Window)
    window.showMaximized()            # open maximized with system buttons

    app.exec_()
