# Dependencies
import cv2
import numpy as np
from masked_line1D_patters_lib import *
from render_scene_gif import *

# Constants
filename = "output.mp4"
FRAME_RANGE = list(np.concatenate((np.arange(260, 425), np.arange(530, 770)), axis=0))
COUNTER_MAX_CONSECUTIVE_LOST_LANE_COMPENSATIONS = 3
RENDER_IMAGE_WIDTH = 600

# Process video
cap = cv2.VideoCapture(filename)
frame_cnt = 0
line_id_last, averaged_lines_last = np.array([]), np.array([])
nxs, nys = 0, 0
xs, ys = [], []

# Iterate over video frames
valid_last_lines, filenames = [], []
while(cap.isOpened()):
    # Log
    print("Frame", frame_cnt)
    
    # Read frame
    _, img = cap.read()
    if frame_cnt == 0:
        img_dim_sum = img.shape[0] + img.shape[1]

    # Check if frame is in frame range
    if FRAME_RANGE != None and frame_cnt not in FRAME_RANGE:
        frame_cnt += 1
        if img is None:
            break
        continue

    # Process image if not corrupt    
    if img is not None:
        # Frame counter
        frame_cnt += 1
        print("1) Frame No.", frame_cnt)

        # Apply filters
        img_canny = pre_process_canny(img)
        img_roi = mask_rectangular_roi(img_canny, color=(255, 255, 255))
        img_roi_orig = cv2.addWeighted(img, 0.4, mask_rectangular_roi(img, color=(255, 255, 255)), 0.7, 1)

        # Detect lines
        lines = cv2.HoughLinesP(img_roi, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

        # Validate lines data, find average parameters
        _, averaged_lines = average_slope_intercept(img_roi, lines)
        line_id, averaged_lines = filter_lines(averaged_lines, img_dim_sum)
        print("2a) Line ID(s):\n", line_id)
        print("3a) Averaged lines:\n", averaged_lines)

        # Collect averaged lines for line ID [1, 2] - left and right lane available
        if np.array_equal(line_id, np.array([1, 2])):
            # Limit gradients
            if len(valid_last_lines) > 0:
                for al in [0, 1]:
                    averaged_line_x = limit_gradient(averaged_lines[al][0], valid_last_lines[-1][al][0])
                    averaged_line_y = limit_gradient(averaged_lines[al][2], valid_last_lines[-1][al][2])
                    averaged_lines[al] = np.array([[averaged_line_x, averaged_lines[al][1], averaged_line_y, averaged_lines[al][3]]])        
            # Append valid lines to storage
            valid_last_lines = append_valid_line_to_storage(averaged_lines, valid_last_lines)
            # Reset lost lane compensation counter
            cnt_lost_lane = 0
        # Else, use stored averaged lines data to compensate lane loss
        elif len(valid_last_lines) > 0 and cnt_lost_lane < COUNTER_MAX_CONSECUTIVE_LOST_LANE_COMPENSATIONS:
            # Determine average coordinates along stored lines
            line_id, averaged_lines = calc_average_coordinates_along_stored_lines(valid_last_lines)
            # Log
            print("Dual lane compensation!")
            print("2a) Line ID(s):\n", line_id)
            print("3a) Averaged lines:\n", averaged_lines)
            # Append valid lines to storage
            valid_last_lines = append_valid_line_to_storage(averaged_lines, valid_last_lines)
            # Increase lost lane compensation counter
            cnt_lost_lane += 1

        # Create visual results
        if line_id.size > 0:
            # print(averaged_lines)
            # img_lines = display_lines(img_roi, averaged_lines, color=line_id2color(line_id))
            img_lines = display_lines(img_roi, averaged_lines, line_id)
            # Filtered footage
            img_final = cv2.addWeighted(img_roi_orig, 0.7, img_lines, 1, 1)
        else:
            print("No lines detected. No lane data offered. No last lanes in storage. Continuing.")
            # Show frame
            # cv2.imshow("Result", img)
            img_final = img_roi_orig
        
        # Resize image
        H, W = img_final.shape[0], img_final.shape[1]
        scaling_factor = RENDER_IMAGE_WIDTH / W
        H, W = int(H * scaling_factor), RENDER_IMAGE_WIDTH
        img_final = cv2.resize(img_final, (W, H))

        # Show results
        cv2.imshow("Result", img_final)

        # Save frame
        filename = 'frames/frame_{0:0>4}.png'.format(frame_cnt)
        cv2.imwrite(filename, img_final)
        filenames.append(filename)

        # Wait between frames
        cv2.waitKey(TIME_WAIT_BETWEEN_FRAMES)
    else:
        break

# Finish video processing and presenting
cap.release()
cv2.destroyAllWindows()

# Render scene as .gif
render_scene_gif(filenames, fps=10)