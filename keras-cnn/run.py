# Set system settings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import dependencies
import numpy as np
import cv2
from keras.models import load_model
from render_scene_gif import *

# Define input, output video filename
input_filename = "input.mp4"
FRAME_RANGE = list(np.arange(800, 900))

# Constants
MAX_AVG_DATA = 15
IMG_SIZE = (1280, 720)
OUTPUT_SIZE = (610, 300)
SMALL_IMG_SIZE = (160, 80)
WAIT_KEY_TIME = 10
DILATE_ERODE_KERNEL = np.ones((4, 4))

# Load model
model = load_model('model.h5')

# Init video stream
cap = cv2.VideoCapture(input_filename)
# Init frame counter
frame_cnt = 0
# Init frame name array
filenames, last_predictions = [], []

# Iterate over all frames
while cap.isOpened():
    # Load current frame
    ret, img = cap.read()
    # If loaded image is corrupt, stop process
    if not ret:
        break

    # Log
    print("Frame", frame_cnt)

    # Resize input img
    img = cv2.resize(img, IMG_SIZE)

    # Use frame range
    if frame_cnt not in FRAME_RANGE:
        frame_cnt += 1
        continue

    # Reduce image size
    small_img = cv2.resize(img, SMALL_IMG_SIZE)
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]

    # Perform prediction and revert normalization
    prediction = model.predict(small_img)[0] * 255
    # Add current prediction to lanes object
    last_predictions.append(prediction)
    # Use last lane fits up to MAX_AVG_DATA
    if len(last_predictions) > MAX_AVG_DATA:
        last_predictions = last_predictions[1:]

    # Calculate average prediction results
    avg_prediction = np.mean(np.array([i for i in last_predictions]), axis=0)
    # Improve results
    avg_prediction = cv2.dilate(avg_prediction, DILATE_ERODE_KERNEL, iterations=1)
    avg_prediction = cv2.threshold(avg_prediction, 25, 255, cv2.THRESH_BINARY)[1]
    avg_prediction = cv2.erode(avg_prediction, DILATE_ERODE_KERNEL, iterations=3)

    # Create empty BGR image, apply prediction results as color channel
    blank_img = np.zeros_like(avg_prediction).astype(np.uint8)
    lane_img = np.dstack((avg_prediction, blank_img, blank_img))
    # Resize resulting lane image
    lane_img = np.asarray(cv2.resize(lane_img, IMG_SIZE), np.uint8)
    # Add input image and prediction results to fill blue and red channel with input data
    img_final = cv2.addWeighted(img, 1, lane_img, 0.8, 0)
    # Resize output image
    img_final = cv2.resize(img_final, OUTPUT_SIZE)

    # Show results
    # cv2.imshow("Result", lane_img)
    cv2.imshow("Result", img_final)
    cv2.waitKey(WAIT_KEY_TIME)

    # Save resulting frame
    filename = 'frames/frame_{0:0>4}.png'.format(frame_cnt)
    cv2.imwrite(filename, img_final)
    filenames.append(filename)

    # Increase frame counter
    frame_cnt += 1


# Release video stream
cap.release()
# Close all windows
cv2.destroyAllWindows()

# Render scene as .gif
render_scene_gif(filenames, fps=15)