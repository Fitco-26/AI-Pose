import cv2
import mediapipe as mp
import numpy as np

#drawings for our body pose
mp_drawing = mp.solutions.drawing_utils
#importing pose estimation models from mediapipe
mp_pose = mp.solutions.pose   
