# 🏋️‍♂️ Fitness Voice Assistant

An AI-powered fitness assistant that helps you track workouts in real-time using **YOLO object detection**, **Mediapipe pose estimation**, and a **Flask web interface**.  
The app counts your curls, tracks progress, and gives voice guidance.

---

## ✨ Features

- 🎥 **Live camera feed** with dumbbell detection (YOLOv8)
- 🧍 **Pose detection & rep counting** (Mediapipe)
- 📊 **Real-time stats panel** (reps, stage, progress bar)
- 🗣️ **Text-to-Speech voice feedback** (pyttsx3)
- 🖥️ **Web-based UI** built with Flask and vanilla JavaScript
- 📦 Easy setup with `requirements.txt`

---

## ⚡ Installation

### 1. Clone Repository

```bash
git clone https://github.com/Fitco-26/AI-Pose.git
cd AI-Pose

# Install Dependencies
pip install -r requirements.txt

### Windows Users:
If you encounter an `ImportError` related to `_framework_bindings` or `DLL load failed`, you need to install the Microsoft C++ Redistributable. Download and run the installer for `X64` under the "Visual Studio 2015, 2017, 2019, and 2022" section.

# Start the app:
python app.py
```
