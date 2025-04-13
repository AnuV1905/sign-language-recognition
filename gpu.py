# GPU testng section
# import tensorflow as tf

# print("TensorFlow version:", tf.__version__)
# print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
# print("GPU Devices:", tf.config.list_physical_devices('GPU'))
# #################################

# cv testing section
import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Webcam accessed successfully!")
else:
    print("Failed to access webcam.")
cap.release()

