# Set system settings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import dependencies
import numpy as np
import cv2
from keras.models import load_model

# Constants
filename = 'Road_detection_FCN/data_road/testing/image_2/uu_000013.png'
SMALL_IMG_SIZE = (160, 80)
OUTPUT_SIZE = (1280, 720)
model = load_model('model.h5')

# Load image
img = cv2.imread(filename, cv2.IMREAD_COLOR)
img_orig = img.astype(np.uint8)
img = cv2.resize(img, SMALL_IMG_SIZE)
img = np.array(img).astype(np.uint8)
img = img[None, :, :, :]

# Perform prediction
pred = model.predict(img)[0] * 255
blank = np.zeros_like(pred)
pred = np.dstack((pred, blank, blank)).astype(np.uint8)
pred = cv2.resize(pred, OUTPUT_SIZE)

# Show final results
img_orig = cv2.resize(img_orig, OUTPUT_SIZE)
img_final = cv2.addWeighted(img_orig, 1, pred, 1, 0)
cv2.imshow("", img_final)
cv2.waitKey(0)