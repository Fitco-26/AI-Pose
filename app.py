import cv2
from ultralytics import YOLO
from modules.curl_counter import CurlCounter
from modules.pose_detector import PoseDetector


def main():
    # Webcam
    cap = cv2.VideoCapture(0)

    # Load Pose + Curl Counter
    detector = PoseDetector()
    counter = CurlCounter()

    # Load trained YOLO model for dumbbell detection
    yolo_model = YOLO("models/best.pt")   # <-- keep your trained model here

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------- YOLO Dumbbell Detection ----------------
        yolo_results = yolo_model(frame)
        dumbbell_detected = False

        for result in yolo_results:
            frame = result.plot()   # overlay bounding boxes

            if len(result.boxes) > 0:  # if YOLO finds any object
                dumbbell_detected = True

        # ---------------- Warning Message if No Dumbbell ----------------
        if not dumbbell_detected:
            cv2.putText(frame,
                        "⚠ Please pick up your dumbbell!",
                        (25, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255), 3)

        # ---------------- Pose Detection ----------------
        image, results = detector.detect(frame)

        if results.pose_landmarks:
            counter.process(image, results.pose_landmarks.landmark)

        detector.draw_landmarks(image, results)

        # ---------------- Show Output ----------------
        cv2.imshow('Fitness Assistant', image)

        # Exit on "q"
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
