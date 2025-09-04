import cv2
from modules.curl_counter import CurlCounter
from modules.pose_detector import PoseDetector


def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    counter = CurlCounter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, results = detector.detect(frame)

        if results.pose_landmarks:
            counter.process(image, results.pose_landmarks.landmark)

        detector.draw_landmarks(image, results)

        cv2.imshow('Mediapipe Curl Counter', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
