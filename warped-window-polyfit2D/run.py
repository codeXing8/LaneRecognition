# Import dependencies
import cv2
import numpy as np
from warped_window_polyfit2D_lib import *
from render_scene_gif import *

# Constants
PICKLE_FILE = 'calibration_pickle.p'
RENDER_IMAGE_WIDTH = 600

# Define video file to load
video_file = 'sample.mp4'
frame_width, frame_height = 1200, 640
FRAME_RANGE = list(np.arange(550, 750))

# Image warp parameters
WARP_POINTS = np.float32([(42 / 100, 63 / 100), (1 - (42 / 100), 63 / 100), (14 / 100, 87 / 100), (1 - (14 / 100), 87 / 100)])
ORIG_POINTS = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])

# Init video stream
cap = cv2.VideoCapture(video_file)
# Init frame counter
frame_cnt = 0
# Init frame name array
filenames = []

# Iterate over all frames
while cap.isOpened():
    # Load current frame
    ret, img = cap.read()

    # If loaded image is corrupt, stop process
    if ret == False:
        break

    # Log
    print("Frame", frame_cnt)
    
    # Use frame range
    if frame_cnt not in FRAME_RANGE:
        frame_cnt += 1
        continue
        
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

    # Resize image
    H, W = img_final.shape[0], img_final.shape[1]
    scaling_factor = RENDER_IMAGE_WIDTH / W
    H, W = int(H * scaling_factor), RENDER_IMAGE_WIDTH
    img_final = cv2.resize(img_final, (W, H))

    # Show results
    cv2.imshow("Results", img_final)
    # Wait
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Save frame
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