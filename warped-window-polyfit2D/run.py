# Import dependencies
import cv2
import numpy as np
from warped_window_polyfit2D_lib import *

# Constants
PICKLE_FILE = 'calibration_pickle.p'

# Define video file to load
video_file = 'sample.mp4'
frame_width, frame_height = 1280, 720

# Image warp parameters
WARP_POINTS = np.float32([(42 / 100, 63 / 100), (1 - (42 / 100), 63 / 100), (14 / 100, 87 / 100), (1 - (14 / 100), 87 / 100)])
ORIG_POINTS = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])

# Init video stream
cap = cv2.VideoCapture(video_file)

# Iterate over all frames
while cap.isOpened():
    # Load current frame
    _, img = cap.read()
    
    # Resize image
    img = cv2.resize(img, (frame_width, frame_height), None)
    
    # Create image copies
    # imgWarpPoints = img.copy()
    img_final = img.copy()
    # img_canny = img.copy()

    # Remove distortion from image
    img_undistorted = undistort_with_cal_pickle_file(img, PICKLE_FILE)
    
    # Run Canny edge detector in a color masked image
    img_masked_canny, img_canny, img_masked = run_masked_canny_detector(img_undistorted)
    
    # Warp perspective to bird's-eye-view
    img_warped = perform_image_warp(img_masked, dst_size=(frame_width, frame_height), src=WARP_POINTS, dst=ORIG_POINTS)

    # Apply sliding window filter to warped image
    img_sliding, curves, _ = apply_sliding_window_filter(img_warped, draw_windows=True)
    # cv2.imshow("Results", img_sliding)

    # Draw lanes on final image
    if curves != None:
        img_final = draw_lanes(img, curves[0], curves[1], frame_width, frame_height, src=[42, 63, 14, 87])
    else:
        img_final = img

    # Show results
    cv2.imshow("Results", img_final)
    # Wait
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video stream
cap.release()
# Close all windows
cv2.destroyAllWindows()