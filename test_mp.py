import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Create a dummy image
img = np.zeros((480, 640, 3), dtype=np.uint8)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

# Detect hands
detection_result = detector.detect(mp_image)
print(f"Detected {len(detection_result.hand_landmarks)} hands")
print("Successfully loaded HandLandmarker!")
