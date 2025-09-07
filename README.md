# 🏋️‍♂️ Fitness Voice Assistant  

An AI-powered fitness assistant that helps you track workouts in real-time using **YOLO object detection**, **Mediapipe pose estimation**, and **PyQt5 GUI**.  
The app counts your curls, tracks progress, and gives voice guidance.  

---

## ✨ Features
- 🎥 **Live camera feed** with dumbbell detection (YOLOv8)  
- 🧍 **Pose detection & rep counting** (Mediapipe)  
- 📊 **Workout stats panel** (reps, stage, progress bar)  
- 🗣️ **Voice assistant integration** (Vosk, gTTS, pyttsx3, faster-whisper)  
- 🖥️ **Cross-platform GUI** built with PyQt5  
- 📦 Easy setup with `requirements.txt` or Docker  

---



## ⚡ Installation  

### 1. Clone Repository  
```bash
git clone https://github.com/your-username/fitness_voice_assistant.git
cd fitness_voice_assistant

# Install Dependencies
pip install -r requirements.txt

# Start the app:
python app.py

